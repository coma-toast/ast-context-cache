package db

import (
	"log"
	"sync/atomic"
)

var checkpointAbort atomic.Bool

// RequestShutdown aborts in-flight WAL maintenance so the process can exit promptly.
func RequestShutdown() {
	checkpointAbort.Store(true)
	if WALMaintenanceActive() {
		log.Println("shutdown: aborting WAL checkpoint")
	}
	if AfterForceCheckpoint != nil {
		AfterForceCheckpoint()
	}
	endWALMaintenance()
}
