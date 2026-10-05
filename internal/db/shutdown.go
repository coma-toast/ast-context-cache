package db

import "sync/atomic"

var checkpointAbort atomic.Bool

// RequestShutdown aborts in-flight WAL maintenance so the process can exit promptly.
func RequestShutdown() {
	checkpointAbort.Store(true)
	if WALMaintenanceActive() {
		logger.Info("Aborting WAL checkpoint for shutdown")
	}
	if AfterForceCheckpoint != nil {
		AfterForceCheckpoint()
	}
	_ = restoreIndexPool()
	endWALMaintenance()
}
