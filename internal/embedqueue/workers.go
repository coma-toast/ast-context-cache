package embedqueue

import (
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	MinWorkers             = 0
	DefaultMaxWorkers      = 15
	AbsoluteMaxWorkers     = 64
	defaultWorkers         = 3
	embedWorkersSetting    = "EMBED_WORKERS"
	embedWorkerMaxSetting  = "embed_worker_max"
	startupProcessingDelay = 10 * time.Second

	// manualOverrideGrace is how long a manual SetWorkerCount/AdjustWorkers (or aux
	// equivalent) call is protected from the WAL-backpressure ceiling. Without this,
	// applyWalBackpressure's 30s ticker would silently reset the pool right back down
	// on its very next tick — embedding activity itself writes to the WAL, so it almost
	// never shrinks by the required 4MB in that window, meaning an operator's manual
	// raise never got a real chance to run before being undone.
	manualOverrideGrace = 2 * time.Minute
)

var errQueueNotStarted = errs.New("embed queue not started")

var (
	workerCount           = defaultWorkers
	workerTarget          = defaultWorkers
	workerLive            atomic.Int64
	workerMu              sync.Mutex
	workerStop            chan struct{}
	startupWorkerOverride *int
	processingReadyAt     atomic.Int64 // UnixNano workers wait for; 0 = none. Polled by running workers.
	lastThrottleApplied   int
	backpressureKick      atomic.Bool
	manualOverrideAt      time.Time
)

func beginProcessingWindow() {
	processingReadyAt.Store(time.Now().Add(startupProcessingDelay).UnixNano())
	if startupProcessingDelay > 0 {
		logger.Info("Embed queue processing delayed", "delay", startupProcessingDelay)
	}
}

func waitForProcessingReady() {
	for {
		at := processingReadyAt.Load()
		if at == 0 {
			return
		}
		d := time.Until(time.Unix(0, at))
		if d <= 0 {
			return
		}
		if d > 50*time.Millisecond {
			d = 50 * time.Millisecond
		}
		time.Sleep(d)
	}
}

// SetStartupWorkers overrides the DB worker count for this process only (not persisted).
func SetStartupWorkers(n int) {
	startupWorkerOverride = &n
}

// WorkerLive returns goroutines still running (may exceed WorkerCount while draining).
func WorkerLive() int {
	return int(workerLive.Load())
}

func notifyWorkerPoolLiveChange() {
	realtime.Notify(realtime.IndexHealth | realtime.HealthBar)
}

// MaxWorkers returns the configured upper limit for embed worker goroutines.
func MaxWorkers() int {
	raw := db.GetSetting(embedWorkerMaxSetting, strconv.Itoa(DefaultMaxWorkers))
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return DefaultMaxWorkers
	}
	if n > AbsoluteMaxWorkers {
		return AbsoluteMaxWorkers
	}
	return n
}

func loadWorkerCount() int {
	if startupWorkerOverride != nil {
		return clampWorkerCount(*startupWorkerOverride)
	}
	raw := db.GetSetting(embedWorkersSetting, strconv.Itoa(defaultWorkers))
	n, err := strconv.Atoi(raw)
	if err != nil || n < MinWorkers {
		return defaultWorkers
	}
	return clampWorkerCount(n)
}

func clampWorkerCount(n int) int {
	if n < MinWorkers {
		n = MinWorkers
	}
	max := MaxWorkers()
	if n > max {
		return max
	}
	return n
}

func persistWorkerCount(n int) {
	if err := db.SetSetting(embedWorkersSetting, strconv.Itoa(n)); err != nil {
		logger.Warn("Failed to persist workers", "error", err)
	}
}

// WorkerCount returns configured embed worker goroutines.
func WorkerCount() int {
	workerMu.Lock()
	defer workerMu.Unlock()
	return workerCount
}

// WorkerTarget returns the persisted/desired worker count (may exceed live count under WAL throttle).
func WorkerTarget() int {
	workerMu.Lock()
	defer workerMu.Unlock()
	return workerTarget
}

func applyWorkerCountLocked(n int, persist bool) error {
	if workerStop == nil {
		return errQueueNotStarted
	}
	for workerCount < n {
		go worker()
		workerCount++
	}
	for workerCount > n {
		workerStop <- struct{}{}
		workerCount--
	}
	if persist {
		workerTarget = workerCount
		persistWorkerCount(workerCount)
		logger.Info("Workers set", "workers", workerCount)
		maybeQuietOnWorkersPaused(workerCount)
		manualOverrideAt = time.Now()
	}
	realtime.Notify(realtime.EmbedFinished | realtime.IndexHealth)
	return nil
}

// ClampWorkersToMax lowers the live worker count when the configured max shrinks.
func ClampWorkersToMax() error {
	workerMu.Lock()
	defer workerMu.Unlock()
	max := MaxWorkers()
	if workerCount <= max {
		return nil
	}
	return applyWorkerCountLocked(max, true)
}

// SetWorkerCount changes the worker pool size (clamped to MinWorkers..MaxWorkers()).
func SetWorkerCount(n int) (int, error) {
	max := MaxWorkers()
	if n < MinWorkers || n > max {
		return WorkerTarget(), workerRangeErr("workers", n, max)
	}
	workerMu.Lock()
	defer workerMu.Unlock()
	if err := applyWorkerCountLocked(n, true); err != nil {
		return workerTarget, err
	}
	return workerTarget, nil
}

// AdjustWorkers atomically adds delta to the persisted worker target.
func AdjustWorkers(delta int) (int, error) {
	workerMu.Lock()
	defer workerMu.Unlock()
	n := workerTarget + delta
	max := MaxWorkers()
	if n < MinWorkers || n > max {
		return workerTarget, workerRangeErr("workers", n, max)
	}
	if err := applyWorkerCountLocked(n, true); err != nil {
		return workerTarget, err
	}
	return workerTarget, nil
}

func workersStarted() bool {
	workerMu.Lock()
	defer workerMu.Unlock()
	return workerStop != nil
}

func startPressureBackoff() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			applyWalBackpressure()
		}
	}()
}

// applyWalBackpressure resizes both embed pools to the current WAL ceiling. The ceiling
// ratchets down to 0 while the WAL refuses to drain, so writers stop and TRUNCATE can win.
// A pool that was manually set within manualOverrideGrace is exempt from being reduced
// below what was requested — see applyPrimaryCeiling/applyAuxCeiling.
func applyWalBackpressure() {
	workerMu.Lock()
	paused := swapPauseDepth > 0
	target := workerTarget
	workerMu.Unlock()
	if paused {
		return
	}
	ceiling := backpressureCeiling(target)
	applyPrimaryCeiling(target, ceiling)
	applyAuxCeiling(ceiling)
	if ceiling == 0 {
		kickBackpressureCheckpoint()
	}
}

// backpressureCeiling is the embed-worker ceiling from WAL size, tightened by low free
// space on the data volume (critical -> 0, which also kicks a WAL TRUNCATE to give space
// back). -1 means no ceiling.
func backpressureCeiling(target int) int {
	return db.DiskSpaceCeiling(db.UpdateWalBackpressure(max(target, AuxWorkerTarget())))
}

func applyPrimaryCeiling(target, ceiling int) {
	n := db.ThrottledEmbedWorkers(target)
	workerMu.Lock()
	if swapPauseDepth > 0 {
		workerMu.Unlock()
		return
	}
	if ceiling >= 0 && ceiling < n && time.Since(manualOverrideAt) >= manualOverrideGrace {
		n = ceiling
	}
	cur := workerCount
	if n == cur {
		lastThrottleApplied = n
		workerMu.Unlock()
		return
	}
	err := applyWorkerCountLocked(n, false)
	got := workerCount
	workerMu.Unlock()
	if err != nil {
		return
	}
	prev := lastThrottleApplied
	lastThrottleApplied = got
	if n < target {
		logger.Info("Throttled workers", "from", cur, "to", n, "target", target, "wal", db.FormatFileSize(db.WalFileBytes()))
	} else if got > prev || (got == target && cur < target) {
		logger.Info("Restored workers", "workers", got, "wal", db.FormatFileSize(db.WalFileBytes()))
	}
}

// applyAuxCeiling throttles the aux pool too: it writes to the same index.db, so leaving it
// running would keep the WAL growing after the primary pool is drained.
func applyAuxCeiling(ceiling int) {
	if MaintenancePaused() {
		return
	}
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	if auxWorkerStop == nil || maintenanceAuxDepth > 0 {
		return
	}
	want := auxWorkerTarget
	if ceiling >= 0 && ceiling < want && time.Since(auxManualOverrideAt) >= manualOverrideGrace {
		want = ceiling
	}
	prev := auxWorkerCount
	if want == prev {
		return
	}
	if err := applyAuxWorkerCountLocked(want, false); err != nil {
		logger.Warn("Failed to apply aux WAL throttle", "error", err)
		return
	}
	logger.Info("Aux workers adjusted for WAL ceiling", "from", prev, "to", want, "target", auxWorkerTarget)
}

// kickBackpressureCheckpoint asks for a forced TRUNCATE once the drained pools go quiet.
// A pending backlog keeps QueueIdleForWAL false, so the quiet-period loop never fires here.
func kickBackpressureCheckpoint() {
	if !backpressureKick.CompareAndSwap(false, true) {
		return
	}
	go func() {
		defer backpressureKick.Store(false)
		deadline := time.Now().Add(2 * time.Minute)
		for time.Now().Before(deadline) {
			if InFlight() == 0 && WorkerLive() == 0 && AuxWorkerLive() == 0 {
				db.TryQuietWALTruncate("wal_backpressure")
				return
			}
			time.Sleep(500 * time.Millisecond)
		}
	}()
}

func workerRangeErr(pool string, n, max int) error {
	return errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("%s must be %d–%d", pool, MinWorkers, max), "workers", n, "max", max)
}
