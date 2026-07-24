package embedqueue

import (
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// stuckRecoverAfter is how long workers must stay stuck before auto-recover.
// Tests may shorten this.
var stuckRecoverAfter = 5 * time.Minute

var (
	stuckWatchOnce    sync.Once
	stuckMu           sync.Mutex
	stuckSince        time.Time
	lastAutoRecoverAt atomic.Int64
)

func startStuckWorkerWatchdog() {
	stuckWatchOnce.Do(func() {
		go stuckWorkerWatchdogLoop()
	})
}

func stuckWorkerWatchdogLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		maybeRecoverStuckWorkers()
	}
}

func resetStuckClock() {
	stuckMu.Lock()
	stuckSince = time.Time{}
	stuckMu.Unlock()
}

// maybeRecoverStuckWorkers restores primary (and aux) workers when the effective
// pool is stuck at zero while the target remains > 0 for stuckRecoverAfter.
// Intentional pause (target 0) and active maintenance/swap are ignored.
func maybeRecoverStuckWorkers() {
	if MaintenancePaused() {
		resetStuckClock()
		return
	}
	if WorkerTarget() <= 0 || WorkerCount() != 0 || WorkerLive() != 0 {
		resetStuckClock()
		return
	}

	stuckMu.Lock()
	if stuckSince.IsZero() {
		stuckSince = time.Now()
		stuckMu.Unlock()
		return
	}
	elapsed := time.Since(stuckSince)
	stuckMu.Unlock()
	if elapsed < stuckRecoverAfter {
		return
	}

	workerMu.Lock()
	target := workerTarget
	n := db.ThrottledEmbedWorkers(target)
	err := applyWorkerCountLocked(n, false)
	workerMu.Unlock()
	if err != nil {
		log.Printf("embedqueue: auto-recover workers failed: %v", err)
		return
	}

	auxTarget := AuxWorkerTarget()
	if auxTarget > 0 && AuxWorkerCount() == 0 {
		auxWorkerMu.Lock()
		if auxWorkerTarget > 0 && auxWorkerCount == 0 {
			if err := applyAuxWorkerCountLocked(auxWorkerTarget, false); err != nil {
				log.Printf("embedqueue: auto-recover aux workers failed: %v", err)
			}
		}
		auxWorkerMu.Unlock()
	}

	lastAutoRecoverAt.Store(time.Now().Unix())
	resetStuckClock()
	log.Printf("embedqueue: auto-recovered workers after stuck pause (target=%d aux=%d)", target, auxTarget)
}

// LastAutoRecoverAt returns the unix time of the last stuck-worker auto-recover, or 0.
func LastAutoRecoverAt() int64 {
	return lastAutoRecoverAt.Load()
}
