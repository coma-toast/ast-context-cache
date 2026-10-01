package embedqueue

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestMaybeQuietOnWorkersPausedNoopWhenNonZero(t *testing.T) {
	maybeQuietOnWorkersPaused(3) // must not block or panic
}

func TestRunQuietPeriodDoesNotPanic(t *testing.T) {
	runQuietPeriod("test")
}

// The quiet-on-pause goroutine pauses only aux, so when it finishes it must
// not undo a swap pause it never took: SetWorkerCount(0) during an embedder
// swap used to end the swap's pause early and restart its saved workers
// against the embedder being swapped out.
func TestQuietOnWorkersPausedKeepsSwapPause(t *testing.T) {
	Start(stubEmbedder{})
	resetPauseStateForTest()
	held := false
	release := func() {
		if held {
			atomic.AddInt64(&inFlight, -1)
			held = false
		}
	}
	t.Cleanup(func() {
		release()
		SetWorkerCount(0)
		resetPauseStateForTest()
	})
	if _, err := SetWorkerCount(2); err != nil {
		t.Fatal(err)
	}
	PrepareForEmbedderSwap(5 * time.Second)

	// Hold the quiet goroutine in its idle wait until it has paused aux, so its
	// restore is observable as maintenanceAuxDepth returning to 0.
	atomic.AddInt64(&inFlight, 1)
	held = true
	if _, err := SetWorkerCount(0); err != nil {
		t.Fatal(err)
	}
	waitMaintenanceAuxDepth(t, 1)
	release()
	waitMaintenanceAuxDepth(t, 0)

	if !SwapPaused() {
		t.Fatal("quiet-on-pause ended the embedder swap's pause")
	}
	if n := WorkerCount(); n != 0 {
		t.Fatalf("WorkerCount() = %d mid-swap, want 0", n)
	}
	RestoreWorkersAfterSwap()
	if SwapPaused() {
		t.Fatal("still swap-paused after the swap's own restore")
	}
}

func waitMaintenanceAuxDepth(t *testing.T, want int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		auxWorkerMu.Lock()
		got := maintenanceAuxDepth
		auxWorkerMu.Unlock()
		if got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("maintenanceAuxDepth = %d, want %d", got, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
