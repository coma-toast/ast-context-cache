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

func TestPauseAuxForMaintenanceUsesTarget(t *testing.T) {
	auxWorkerMu.Lock()
	prevCount, prevTarget, prevDepth, prevRestore := auxWorkerCount, auxWorkerTarget, maintenanceAuxDepth, maintenanceRestoreAux
	auxWorkerStop = make(chan struct{}, 8)
	auxWorkerCount = 2
	auxWorkerTarget = 5
	maintenanceAuxDepth = 0
	maintenanceRestoreAux = 0
	auxWorkerMu.Unlock()
	defer func() {
		auxWorkerMu.Lock()
		// Drain any stop signals from pause.
		for len(auxWorkerStop) > 0 {
			<-auxWorkerStop
		}
		auxWorkerCount, auxWorkerTarget = prevCount, prevTarget
		maintenanceAuxDepth, maintenanceRestoreAux = prevDepth, prevRestore
		auxWorkerStop = nil
		auxWorkerMu.Unlock()
	}()

	pauseAuxForMaintenance()
	auxWorkerMu.Lock()
	gotCount, gotTarget, gotRestore, gotDepth := auxWorkerCount, auxWorkerTarget, maintenanceRestoreAux, maintenanceAuxDepth
	auxWorkerMu.Unlock()
	if gotCount != 0 {
		t.Fatalf("aux live count=%d want 0", gotCount)
	}
	if gotTarget != 5 {
		t.Fatalf("aux target changed to %d; want unchanged 5", gotTarget)
	}
	if gotRestore != 5 {
		t.Fatalf("restore aux=%d want 5 (saved target)", gotRestore)
	}
	if gotDepth != 1 {
		t.Fatalf("maintenanceAuxDepth=%d want 1", gotDepth)
	}
}
