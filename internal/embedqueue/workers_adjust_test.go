package embedqueue

import "testing"

func TestAdjustWorkersUsesTargetWhenThrottled(t *testing.T) {
	Start(stubEmbedder{})
	resetPauseStateForTest()
	t.Cleanup(func() {
		resetPauseStateForTest()
		_, _ = SetWorkerCount(0)
		waitLiveZero(t, &workerLive)
	})
	// Target 10, live pool held at 4 (the WAL throttle's shape). Through
	// applyWorkerCountLocked rather than assigning workerStop/workerCount, which
	// running workers read and Start's pool has to keep matching.
	workerMu.Lock()
	workerTarget = 10
	err := applyWorkerCountLocked(4, false)
	workerMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	n, err := AdjustWorkers(1)
	if err != nil {
		t.Fatal(err)
	}
	if n != 11 {
		t.Fatalf("target=%d want 11", n)
	}
	if got := WorkerTarget(); got != 11 {
		t.Fatalf("WorkerTarget=%d want 11", got)
	}
}
