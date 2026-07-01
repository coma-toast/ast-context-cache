package db

import (
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
	defer func() { WALInFlightHook = nil }()
	_, ready := prepCheckpoint(true)
	if ready {
		t.Fatal("expected prepCheckpoint to abort when in-flight > 0")
	}
}
