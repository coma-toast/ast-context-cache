package embedqueue

import (
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
		logger.Warn("Maintenance drain timed out with in-flight embeds", "in_flight", n, "primary_live", WorkerLive(), "aux_live", AuxWorkerLive())
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
		logger.Warn("Failed to pause aux workers for maintenance", "error", err)
		return
	}
	logger.Info("Paused aux workers for DB maintenance", "workers", prev, "target", maintenanceRestoreAux)
}

// RestoreAfterMaintenance resumes workers paused by PauseAllForMaintenance.
func RestoreAfterMaintenance() {
	RestoreWorkersAfterSwap()
	restoreAuxAfterMaintenance()
}

// restoreAuxAfterMaintenance undoes one pauseAuxForMaintenance. A caller that
// paused only aux uses it rather than RestoreAfterMaintenance, which would also
// undo a swap pause that caller never took.
func restoreAuxAfterMaintenance() {
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
		logger.Warn("Failed to restore aux workers after maintenance", "error", err)
		return
	}
	logger.Info("Restored aux workers after DB maintenance", "workers", n)
}
