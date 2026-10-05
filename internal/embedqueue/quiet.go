package embedqueue

import (
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
)

const (
	quietIdlePoll    = time.Minute
	quietIdleSustain = 2 * time.Minute
)

var (
	// quietOnPause counts maybeQuietOnWorkersPaused's goroutines, which can wait
	// up to 30s for a quiet window; closing quietOnPauseStop ends that wait.
	quietOnPause     sync.WaitGroup
	quietOnPauseMu   sync.Mutex
	quietOnPauseStop = make(chan struct{})
)

func runQuietPeriod(reason string) {
	db.TryQuietWALTruncate(reason)
	docs.TryQuietRefresh(reason)
}

// StartQuietPeriodLoop watches for sustained embed-queue idle and runs quiet maintenance
// (forced WAL TRUNCATE when large + stale fetch_doc cache refresh).
func StartQuietPeriodLoop() {
	go func() {
		ticker := time.NewTicker(quietIdlePoll)
		defer ticker.Stop()
		var idleSince time.Time
		for range ticker.C {
			if !QueueIdleForWAL() {
				idleSince = time.Time{}
				continue
			}
			if idleSince.IsZero() {
				idleSince = time.Now()
				continue
			}
			if time.Since(idleSince) < quietIdleSustain {
				continue
			}
			logger.Info("Quiet period sustained; running maintenance", "idle", quietIdleSustain)
			runQuietPeriod("queue_idle")
			// Restart sustain window; WAL/docs cooldowns gate actual work.
			idleSince = time.Now()
		}
	}()
}

// maybeQuietOnWorkersPaused runs quiet maintenance when the primary worker target hits 0.
// Aux is paused too so it cannot keep writing while we wait for a quiet WAL window.
// Only that aux pause is undone afterwards: an embedder swap may hold its own.
func maybeQuietOnWorkersPaused(n int) {
	if n != 0 {
		return
	}
	quietOnPauseMu.Lock()
	stop := quietOnPauseStop
	quietOnPause.Add(1)
	quietOnPauseMu.Unlock()
	go func() {
		defer quietOnPause.Done()
		pauseAuxForMaintenance()
		defer restoreAuxAfterMaintenance()
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			if QueueIdleForWAL() && WorkerLive() == 0 && AuxWorkerLive() == 0 {
				runQuietPeriod("workers_paused")
				return
			}
			select {
			case <-stop:
				return
			case <-time.After(200 * time.Millisecond):
			}
		}
		runQuietPeriod("workers_paused")
	}()
}

// stopQuietOnPause ends every quiet-on-pause wait early, skipping its quiet
// period, and waits for each to restore the aux workers it paused. Tests call
// it so one test's SetWorkerCount(0) can't pause aux workers under the next.
// Not with workerMu or auxWorkerMu held: the wait loop's QueueIdleForWAL takes
// both, and the restore takes auxWorkerMu.
func stopQuietOnPause() {
	quietOnPauseMu.Lock()
	close(quietOnPauseStop)
	quietOnPauseStop = make(chan struct{})
	quietOnPauseMu.Unlock()
	quietOnPause.Wait()
}
