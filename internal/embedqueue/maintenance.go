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
func PauseAllForMaintenance(timeout time.Duration) {
	PrepareForEmbedderSwap(timeout)
	auxWorkerMu.Lock()
	maintenanceAuxDepth++
	if maintenanceAuxDepth == 1 {
		maintenanceRestoreAux = auxWorkerCount
		if auxWorkerCount > 0 {
			if err := applyAuxWorkerCountLocked(0, false); err != nil {
				log.Printf("embedqueue: maintenance pause aux workers: %v", err)
			} else {
				log.Printf("embedqueue: paused %d aux workers for DB maintenance", maintenanceRestoreAux)
			}
		}
	}
	auxWorkerMu.Unlock()
	cancelInFlightEmbedderRequests(queueAuxEmbedder())
	deadline := time.Now().Add(timeout)
	for atomic.LoadInt64(&inFlight) > 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
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
