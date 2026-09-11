package embedqueue

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// TestApplyPrimaryCeilingRespectsRecentManualOverride reproduces the reported bug: an
// operator manually raises workers, and within seconds the WAL-backpressure ticker
// silently resets them back down to a stuck ceiling before the change had any real
// chance to run. A manual change must hold for manualOverrideGrace before the ceiling
// is allowed to reduce it again.
//
// Start()/SetWorkerCount (not direct package-var pokes) are used throughout, matching
// the pattern in watchdog_test.go: Start()'s worker goroutines and workerStop channel
// are process-wide (guarded by sync.Once) and shared with every other test in this
// package's binary, so replacing workerStop or setting workerCount without spawning
// matching goroutines leaves stale stop-signals that corrupt whichever test runs next.
func TestApplyPrimaryCeilingRespectsRecentManualOverride(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	Start(stubEmbedder{})
	resetPauseStateForTest()
	if _, err := SetWorkerCount(5); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		resetPauseStateForTest()
		_, _ = SetWorkerCount(0)
		waitLiveZero(t, &workerLive)
	})

	workerMu.Lock()
	manualOverrideAt = time.Now() // simulates the operator just calling SetWorkerCount
	workerMu.Unlock()

	// A ceiling of 0 is the worst case (WAL stuck, ratchet fully down) — within the
	// grace window it must NOT reduce the pool the operator just set.
	applyPrimaryCeiling(5, 0)
	if got := WorkerCount(); got != 5 {
		t.Fatalf("worker count = %d during grace window, want unchanged at 5", got)
	}

	// Once the grace window has elapsed, the ceiling must resume enforcing itself.
	workerMu.Lock()
	manualOverrideAt = time.Now().Add(-manualOverrideGrace - time.Second)
	workerMu.Unlock()
	applyPrimaryCeiling(5, 0)
	if got := WorkerCount(); got != 0 {
		t.Fatalf("worker count = %d after grace window elapsed, want ratcheted to 0", got)
	}
	waitLiveZero(t, &workerLive)
}

// TestApplyAuxCeilingRespectsRecentManualOverride mirrors the primary-pool test for the
// aux pool, since a manual aux worker-count change hit the identical bug.
func TestApplyAuxCeilingRespectsRecentManualOverride(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	Start(stubEmbedder{})
	ensureAuxStopForTest()
	resetPauseStateForTest()
	if _, err := SetAuxWorkerCount(3); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		resetPauseStateForTest()
		_, _ = SetAuxWorkerCount(0)
		waitLiveZero(t, &auxWorkerLive)
	})

	auxWorkerMu.Lock()
	auxManualOverrideAt = time.Now()
	auxWorkerMu.Unlock()

	applyAuxCeiling(0)
	if got := AuxWorkerCount(); got != 3 {
		t.Fatalf("aux worker count = %d during grace window, want unchanged at 3", got)
	}

	auxWorkerMu.Lock()
	auxManualOverrideAt = time.Now().Add(-manualOverrideGrace - time.Second)
	auxWorkerMu.Unlock()
	applyAuxCeiling(0)
	if got := AuxWorkerCount(); got != 0 {
		t.Fatalf("aux worker count = %d after grace window elapsed, want ratcheted to 0", got)
	}
	waitLiveZero(t, &auxWorkerLive)
}
