package embedqueue

import (
	"log"
	"sync/atomic"
	"time"
)

var (
	maintenanceRestoreAux int
	maintenanceAuxDepth   int
)

// MaintenancePaused reports whether embed workers are paused for DB maintenance or swap.
func MaintenancePaused() bool {
	if SwapPaused() {
		return true
	}
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	return maintenanceAuxDepth > 0
}

// QueueIdleForWAL is true when no embed work is queued, pending, or in-flight.
func QueueIdleForWAL() bool {
	s := Snapshot()
	pendingQueued := 0
	if pendingCh != nil {
		pendingQueued = len(pendingCh)
	}
	return s.InFlight == 0 && s.Queued == 0 && s.Pending == 0 && pendingQueued == 0
}

// PauseAllForMaintenance stops primary and aux embed workers and waits for in-flight work.
// Aux is paused first so it cannot keep feeding jobs while primary drains.
func PauseAllForMaintenance(timeout time.Duration) {
	if timeout <= 0 {
		timeout = defaultSwapDrainTimeout
	}
	pauseAuxForMaintenance()
	cancelInFlightEmbedderRequests(queueAuxEmbedder())
	PrepareForEmbedderSwap(timeout)
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if atomic.LoadInt64(&inFlight) == 0 && WorkerLive() == 0 && AuxWorkerLive() == 0 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if n := atomic.LoadInt64(&inFlight); n > 0 {
		log.Printf("embedqueue: maintenance drain timed out with %d in-flight embed(s) (primary_live=%d aux_live=%d)", n, WorkerLive(), AuxWorkerLive())
	}
}

func pauseAuxForMaintenance() {
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	maintenanceAuxDepth++
	if maintenanceAuxDepth != 1 {
		return
	}
	// Prefer target so restore brings back the configured pool, not a mid-drain live count.
	maintenanceRestoreAux = auxWorkerTarget
	if auxWorkerStop == nil {
		return
	}
	if auxWorkerCount == 0 && auxWorkerTarget == 0 {
		return
	}
	prev := auxWorkerCount
	if err := applyAuxWorkerCountLocked(0, false); err != nil {
		log.Printf("embedqueue: maintenance pause aux workers: %v", err)
		return
	}
	log.Printf("embedqueue: paused %d aux workers for DB maintenance (target was %d)", prev, maintenanceRestoreAux)
}

// RestoreAfterMaintenance resumes workers paused by PauseAllForMaintenance.
func RestoreAfterMaintenance() {
	RestoreWorkersAfterSwap()
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	if maintenanceAuxDepth <= 0 {
		return
	}
	maintenanceAuxDepth--
	if maintenanceAuxDepth > 0 {
		return
	}
	n := maintenanceRestoreAux
	maintenanceRestoreAux = 0
	if n <= 0 || auxWorkerStop == nil {
		return
	}
	max := AuxMaxWorkers()
	if n > max {
		n = max
	}
	if err := applyAuxWorkerCountLocked(n, false); err != nil {
		log.Printf("embedqueue: restore aux workers after maintenance: %v", err)
		return
	}
	log.Printf("embedqueue: restored %d aux workers after DB maintenance", n)
}
