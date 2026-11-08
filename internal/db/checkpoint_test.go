package db

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestTrackCheckpointResult(t *testing.T) {
	walBusyStreak.Store(0)
	trackCheckpointResult(true, 1, 100)
	trackCheckpointResult(true, 1, 100)
	if got := walBusyStreak.Load(); got != 2 {
		t.Fatalf("busy streak=%d want 2", got)
	}
	trackCheckpointResult(true, 0, 100)
	if got := walBusyStreak.Load(); got != 0 {
		t.Fatalf("busy streak after success=%d want 0", got)
	}
}

func TestMaybeForceCheckpointThreshold(t *testing.T) {
	walBusyStreak.Store(0)
	if shouldForceCheckpoint(walForceBytes - 1) {
		t.Fatal("should not force below walForceBytes without busy streak")
	}
	walBusyStreak.Store(3)
	if !shouldForceCheckpoint(walTruncateBytes) {
		t.Fatal("should force when busy streak >= 3")
	}
	walBusyStreak.Store(0)
	if !shouldForceCheckpoint(walForceBytes) {
		t.Fatal("should force at walForceBytes")
	}
}

func TestTruncateBackoffOnBusy(t *testing.T) {
	ResetMaintBackoffForTest()
	maintBusyCycles = 0
	noteMaintResult(1)
	if maintBusyCycles != 1 {
		t.Fatalf("cycles=%d want 1", maintBusyCycles)
	}
	if time.Until(maintBackoffUntil) < 4*time.Minute {
		t.Fatalf("backoff too short: until=%v", maintBackoffUntil)
	}
	maintBusyCycles = 2
	noteMaintResult(1)
	maintBusyCycles = 2
	noteMaintResult(1)
	if time.Until(maintBackoffUntil) < 14*time.Minute {
		t.Fatalf("expected 15m backoff after 3 busy cycles, until=%v", maintBackoffUntil)
	}
	ResetMaintBackoffForTest()
}

func TestShouldDeferMaint(t *testing.T) {
	ResetMaintBackoffForTest()
	maintBackoffUntil = time.Now().Add(5 * time.Minute)
	if !shouldDeferMaint() {
		t.Fatal("expected defer during backoff")
	}
	ResetMaintBackoffForTest()
	if shouldDeferMaint() {
		t.Fatal("expected no defer after reset")
	}
}

func TestEmbedDeferForcesAfterDeadline(t *testing.T) {
	ResetMaintBackoffForTest()
	EmbedQueueIdleHook = func() bool { return false }
	defer func() { EmbedQueueIdleHook = nil }()

	force, skip := truncateMaintForce(walTruncateBytes + 1)
	if !skip || force {
		t.Fatalf("first call skip=%v force=%v want skip=true force=false", skip, force)
	}
	if TruncateDeferUntilForTest().IsZero() {
		t.Fatal("expected defer deadline")
	}

	SetTruncateDeferUntilForTest(time.Now().Add(-time.Second))
	force, skip = truncateMaintForce(walTruncateBytes + 1)
	if skip || !force {
		t.Fatalf("after deadline skip=%v force=%v want skip=false force=true", skip, force)
	}
}

func TestEmbedDeferClearsWhenIdle(t *testing.T) {
	ResetMaintBackoffForTest()
	SetTruncateDeferUntilForTest(time.Now().Add(time.Minute))
	EmbedQueueIdleHook = func() bool { return true }
	defer func() { EmbedQueueIdleHook = nil }()

	force, skip := truncateMaintForce(walTruncateBytes + 1)
	if skip || force {
		t.Fatalf("idle skip=%v force=%v want skip=false force=false", skip, force)
	}
	if !TruncateDeferUntilForTest().IsZero() {
		t.Fatal("expected defer cleared when idle")
	}
}

func TestPrepCheckpointAbortsWhenInFlight(t *testing.T) {
	ResetMaintBackoffForTest()
	WALInFlightHook = func() int64 { return 2 }
	paused, restored := false, false
	BeforeForceCheckpoint = func() { paused = true }
	AfterForceCheckpoint = func() { restored = true }
	defer func() {
		WALInFlightHook = nil
		BeforeForceCheckpoint = nil
		AfterForceCheckpoint = nil
	}()
	restore, ready := prepCheckpoint(true)
	if ready {
		t.Fatal("expected prepCheckpoint to abort when in-flight > 0")
	}
	if !paused {
		t.Fatal("expected BeforeForceCheckpoint to run before abort")
	}
	if restore == nil {
		t.Fatal("expected restore func")
	}
	restore()
	if !restored {
		t.Fatal("expected AfterForceCheckpoint restore after in-flight abort")
	}
}

func TestShouldAttemptWALOnEmbedderError(t *testing.T) {
	now := time.Now()
	if shouldAttemptWALOnEmbedderError(now, walTruncateBytes-1, false, time.Time{}) {
		t.Fatal("should skip when WAL below truncate threshold")
	}
	if shouldAttemptWALOnEmbedderError(now, walTruncateBytes, true, time.Time{}) {
		t.Fatal("should skip when maintenance already active")
	}
	if !shouldAttemptWALOnEmbedderError(now, walTruncateBytes, false, time.Time{}) {
		t.Fatal("should attempt when WAL large and idle")
	}
	if shouldAttemptWALOnEmbedderError(now, walTruncateBytes, false, now.Add(-time.Minute)) {
		t.Fatal("should skip during cooldown")
	}
	if !shouldAttemptWALOnEmbedderError(now, walTruncateBytes, false, now.Add(-quietWALCooldown-time.Second)) {
		t.Fatal("should attempt after cooldown expires")
	}
}

func TestSIGBUSRepro_StressGetSettingDuringWALMaintenance(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	// Seed some settings to read
	if _, err := DB.Exec(`INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)`, "test.setting", "test-value"); err != nil {
		t.Fatal(err)
	}

	const numGoroutines = 20
	const iterations = 1000

	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	// Hot loop of GetSetting on many goroutines
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				val := GetSetting("test.setting", "default")
				if val != "test-value" {
					errCh <- fmt.Errorf("unexpected value: %s", val)
					return
				}
				// Small yield to increase interleaving chance
				runtime.Gosched()
			}
		}()
	}

	// Force multiple WAL maintenance cycles during the reads
	for cycle := 0; cycle < 10; cycle++ {
		busy, frames, ckpt, err := ForceCheckpointWAL()
		if err != nil {
			errCh <- fmt.Errorf("ForceCheckpointWAL error: %v", err)
			break
		}
		t.Logf("cycle %d: busy=%d frames=%d checkpointed=%d", cycle, busy, frames, ckpt)
		// Small delay between cycles
		time.Sleep(10 * time.Millisecond)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}

func TestSIGBUSRepro_StressGetSettingDuringQuietPeriod(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	// Seed some settings to read
	if _, err := DB.Exec(`INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)`, "test.setting", "test-value"); err != nil {
		t.Fatal(err)
	}

	// Make embed queue appear idle so TryQuietWALTruncate proceeds
	EmbedQueueIdleHook = func() bool { return true }
	defer func() { EmbedQueueIdleHook = nil }()

	const numGoroutines = 20
	const iterations = 1000

	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	// Hot loop of GetSetting on many goroutines
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				val := GetSetting("test.setting", "default")
				if val != "test-value" {
					errCh <- fmt.Errorf("unexpected value: %s", val)
					return
				}
				runtime.Gosched()
			}
		}()
	}

	// Force multiple quiet WAL TRUNCATE cycles during the reads
	for cycle := 0; cycle < 10; cycle++ {
		TryQuietWALTruncate("test")
		// Give the async goroutine time to run
		time.Sleep(20 * time.Millisecond)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}
