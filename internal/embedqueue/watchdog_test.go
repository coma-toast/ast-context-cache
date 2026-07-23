package embedqueue

import (
	"sync/atomic"
	"testing"
	"time"
)

func waitLiveZero(t *testing.T, live *atomic.Int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for live.Load() != 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if live.Load() != 0 {
		t.Fatalf("live workers still %d", live.Load())
	}
}

func ensureAuxStopForTest() {
	auxWorkerMu.Lock()
	if auxWorkerStop == nil {
		auxWorkerStop = make(chan struct{}, absoluteAuxMaxWorkers)
	}
	auxWorkerMu.Unlock()
}

// resetPauseStateForTest clears swap/maintenance pause left by prior tests
// (e.g. maybeQuietOnWorkersPaused after SetWorkerCount(0)).
func resetPauseStateForTest() {
	processingReadyAt = time.Time{}
	workerMu.Lock()
	swapPauseDepth = 0
	swapRestoreWorkers = 0
	workerMu.Unlock()
	auxWorkerMu.Lock()
	maintenanceAuxDepth = 0
	maintenanceRestoreAux = 0
	auxWorkerMu.Unlock()
}

// forceStuckWorkers leaves target > 0 with effective/live at 0 (failed restore shape).
func forceStuckWorkers(t *testing.T, primaryTarget, auxTarget int) {
	t.Helper()
	Start(stubEmbedder{})
	ensureAuxStopForTest()
	resetPauseStateForTest()

	workerMu.Lock()
	workerTarget = primaryTarget
	if err := applyWorkerCountLocked(0, false); err != nil {
		workerMu.Unlock()
		t.Fatal(err)
	}
	workerMu.Unlock()
	waitLiveZero(t, &workerLive)

	auxWorkerMu.Lock()
	auxWorkerTarget = auxTarget
	if auxWorkerStop != nil {
		if err := applyAuxWorkerCountLocked(0, false); err != nil {
			auxWorkerMu.Unlock()
			t.Fatal(err)
		}
	} else {
		auxWorkerCount = 0
	}
	auxWorkerMu.Unlock()
	waitLiveZero(t, &auxWorkerLive)
}

func TestStuckWorkerAutoRecover(t *testing.T) {
	prevAfter := stuckRecoverAfter
	stuckRecoverAfter = 25 * time.Millisecond
	defer func() { stuckRecoverAfter = prevAfter }()

	forceStuckWorkers(t, 3, 2)
	resetStuckClock()
	lastAutoRecoverAt.Store(0)
	t.Cleanup(func() {
		resetStuckClock()
		lastAutoRecoverAt.Store(0)
		resetPauseStateForTest()
		_, _ = SetWorkerCount(0)
		auxWorkerMu.Lock()
		_ = applyAuxWorkerCountLocked(0, false)
		auxWorkerTarget = 0
		auxWorkerMu.Unlock()
		resetPauseStateForTest()
		waitLiveZero(t, &workerLive)
		waitLiveZero(t, &auxWorkerLive)
	})

	maybeRecoverStuckWorkers()
	if got := WorkerCount(); got != 0 {
		t.Fatalf("WorkerCount=%d want 0 before sustain window", got)
	}
	if Snapshot().LastAutoRecoverUnix != 0 {
		t.Fatal("expected no auto-recover yet")
	}

	time.Sleep(40 * time.Millisecond)
	maybeRecoverStuckWorkers()

	if got := WorkerCount(); got != 3 {
		t.Fatalf("WorkerCount=%d want 3 after auto-recover", got)
	}
	if got := AuxWorkerCount(); got != 2 {
		t.Fatalf("AuxWorkerCount=%d want 2 after auto-recover", got)
	}
	if Snapshot().LastAutoRecoverUnix == 0 {
		t.Fatal("expected LastAutoRecoverUnix set")
	}
	if LastAutoRecoverAt() == 0 {
		t.Fatal("expected LastAutoRecoverAt set")
	}
}

func TestStuckWorkerNoRecoverWhenTargetZero(t *testing.T) {
	prevAfter := stuckRecoverAfter
	stuckRecoverAfter = 10 * time.Millisecond
	defer func() { stuckRecoverAfter = prevAfter }()

	Start(stubEmbedder{})
	resetPauseStateForTest()
	if _, err := SetWorkerCount(0); err != nil {
		t.Fatal(err)
	}
	resetPauseStateForTest() // quiet-on-pause may have bumped maintenance depth
	waitLiveZero(t, &workerLive)
	resetStuckClock()
	lastAutoRecoverAt.Store(0)
	t.Cleanup(func() {
		resetStuckClock()
		lastAutoRecoverAt.Store(0)
		resetPauseStateForTest()
	})

	maybeRecoverStuckWorkers()
	time.Sleep(20 * time.Millisecond)
	maybeRecoverStuckWorkers()

	if got := WorkerCount(); got != 0 {
		t.Fatalf("WorkerCount=%d want 0 (intentional pause)", got)
	}
	if Snapshot().LastAutoRecoverUnix != 0 {
		t.Fatal("must not auto-recover intentional workers=0")
	}
}

func TestStuckWorkerNoRecoverDuringMaintenance(t *testing.T) {
	prevAfter := stuckRecoverAfter
	stuckRecoverAfter = 10 * time.Millisecond
	defer func() { stuckRecoverAfter = prevAfter }()

	forceStuckWorkers(t, 4, 0)
	workerMu.Lock()
	prevDepth := swapPauseDepth
	swapPauseDepth = 1
	workerMu.Unlock()
	resetStuckClock()
	lastAutoRecoverAt.Store(0)
	t.Cleanup(func() {
		workerMu.Lock()
		swapPauseDepth = prevDepth
		workerMu.Unlock()
		resetStuckClock()
		lastAutoRecoverAt.Store(0)
		resetPauseStateForTest()
		_, _ = SetWorkerCount(0)
		resetPauseStateForTest()
		waitLiveZero(t, &workerLive)
	})

	maybeRecoverStuckWorkers()
	time.Sleep(20 * time.Millisecond)
	maybeRecoverStuckWorkers()

	if got := WorkerCount(); got != 0 {
		t.Fatalf("WorkerCount=%d want 0 during maintenance", got)
	}
	if Snapshot().LastAutoRecoverUnix != 0 {
		t.Fatal("must not auto-recover during maintenance/swap pause")
	}
}

func TestShouldLogFlushRateLimit(t *testing.T) {
	flushLogMu.Lock()
	lastFlushLogAt = time.Time{}
	lastFlushPending = 0
	lastFlushInFlight = 0
	flushLogMu.Unlock()

	if !shouldLogFlush(5, 2) {
		t.Fatal("first flush log should be allowed")
	}
	if shouldLogFlush(5, 2) {
		t.Fatal("identical flush with inFlight>0 within 10s should be skipped")
	}
	if !shouldLogFlush(6, 2) {
		t.Fatal("pending change should allow log")
	}
	if !shouldLogFlush(6, 0) {
		t.Fatal("inFlight=0 should always allow log")
	}
	if !shouldLogFlush(6, 0) {
		t.Fatal("inFlight=0 again should still allow log")
	}
}
