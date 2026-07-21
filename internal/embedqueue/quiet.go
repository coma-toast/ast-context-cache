package embedqueue

import (
	"log"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
)

const (
	quietIdlePoll    = time.Minute
	quietIdleSustain = 2 * time.Minute
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
			log.Printf("embedqueue: quiet period sustained (%s idle) — running maintenance", quietIdleSustain)
			runQuietPeriod("queue_idle")
			// Restart sustain window; WAL/docs cooldowns gate actual work.
			idleSince = time.Now()
		}
	}()
}

// maybeQuietOnWorkersPaused runs quiet maintenance when the primary worker target hits 0.
func maybeQuietOnWorkersPaused(n int) {
	if n != 0 {
		return
	}
	go func() {
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			if QueueIdleForWAL() && WorkerLive() == 0 {
				runQuietPeriod("workers_paused")
				return
			}
			time.Sleep(200 * time.Millisecond)
		}
		runQuietPeriod("workers_paused")
	}()
}
