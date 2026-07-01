package embedqueue

import (
	"sync/atomic"
	"testing"
)

func TestQueueIdleForWALIncludesPending(t *testing.T) {
	pendingCh = make(chan job, 4)
	pendingMu.Lock()
	pending = map[string]job{"a\x00/p": {file: "a", projectPath: "/p"}}
	pendingMu.Unlock()
	atomic.StoreInt64(&inFlight, 0)
	if QueueIdleForWAL() {
		t.Fatal("expected not idle with pending backlog")
	}
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	if !QueueIdleForWAL() {
		t.Fatal("expected idle with empty pending")
	}
}

func TestFlushPendingBlockedDuringMaintenance(t *testing.T) {
	pendingCh = make(chan job, 4)
	pendingMu.Lock()
	pending = map[string]job{"f\x00/p": {file: "f", projectPath: "/p"}}
	pendingMu.Unlock()
	workerMu.Lock()
	swapPauseDepth = 1
	workerMu.Unlock()
	defer func() {
		workerMu.Lock()
		swapPauseDepth = 0
		workerMu.Unlock()
		pendingMu.Lock()
		pending = map[string]job{}
		pendingMu.Unlock()
	}()
	flushPendingIfReady()
	if len(pendingCh) != 0 {
		t.Fatalf("expected no re-queue during maintenance, pendingCh=%d", len(pendingCh))
	}
}
