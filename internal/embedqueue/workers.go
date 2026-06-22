package embedqueue

import (
	"fmt"
	"log"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	MinWorkers            = 0
	DefaultMaxWorkers     = 15
	AbsoluteMaxWorkers    = 64
	defaultWorkers        = 3
	embedWorkersSetting   = "EMBED_WORKERS"
	embedWorkerMaxSetting = "embed_worker_max"
)

var (
	workerCount   = defaultWorkers
	workerTarget  = defaultWorkers
	workerLive    atomic.Int64
	workerMu    sync.Mutex
	workerStop  chan struct{}
)

// WorkerLive returns goroutines still running (may exceed WorkerCount while draining).
func WorkerLive() int {
	return int(workerLive.Load())
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
	raw := db.GetSetting(embedWorkersSetting, strconv.Itoa(defaultWorkers))
	n, err := strconv.Atoi(raw)
	if err != nil || n < MinWorkers {
		return defaultWorkers
	}
	max := MaxWorkers()
	if n > max {
		return max
	}
	return n
}

func persistWorkerCount(n int) {
	if err := db.SetSetting(embedWorkersSetting, strconv.Itoa(n)); err != nil {
		log.Printf("embedqueue: persist workers: %v", err)
	}
}

// WorkerCount returns configured embed worker goroutines.
func WorkerCount() int {
	workerMu.Lock()
	defer workerMu.Unlock()
	return workerCount
}

func applyWorkerCountLocked(n int, persist bool) error {
	if workerStop == nil {
		return fmt.Errorf("embed queue not started")
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
		log.Printf("embed queue: workers set to %d", workerCount)
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
		return WorkerCount(), fmt.Errorf("workers must be %d–%d", MinWorkers, max)
	}
	workerMu.Lock()
	defer workerMu.Unlock()
	if err := applyWorkerCountLocked(n, true); err != nil {
		return workerCount, err
	}
	return workerCount, nil
}

// AdjustWorkers atomically adds delta to the current worker count.
func AdjustWorkers(delta int) (int, error) {
	workerMu.Lock()
	defer workerMu.Unlock()
	n := workerCount + delta
	max := MaxWorkers()
	if n < MinWorkers || n > max {
		return workerCount, fmt.Errorf("workers must be %d–%d", MinWorkers, max)
	}
	if err := applyWorkerCountLocked(n, true); err != nil {
		return workerCount, err
	}
	return workerCount, nil
}

func startPressureBackoff() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		var lastApplied int
		for range ticker.C {
			workerMu.Lock()
			target := workerTarget
			n := db.ThrottledEmbedWorkers(target)
			workerMu.Unlock()
			if n == lastApplied {
				continue
			}
			workerMu.Lock()
			err := applyWorkerCountLocked(n, false)
			got := workerCount
			workerMu.Unlock()
			if err != nil {
				continue
			}
			lastApplied = got
			if n < target {
				log.Printf("embed queue: throttled workers %d -> %d (wal=%s)", target, n, db.FormatFileSize(db.WalFileBytes()))
			}
		}
	}()
}
