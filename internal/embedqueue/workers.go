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
	MinWorkers         = 0
	MaxWorkers         = 10
	defaultWorkers     = 3
	embedWorkersSetting = "EMBED_WORKERS"
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

func loadWorkerCount() int {
	raw := db.GetSetting(embedWorkersSetting, strconv.Itoa(defaultWorkers))
	n, err := strconv.Atoi(raw)
	if err != nil || n < MinWorkers {
		return defaultWorkers
	}
	if n > MaxWorkers {
		return MaxWorkers
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

// SetWorkerCount changes the worker pool size (clamped to MinWorkers..MaxWorkers).
func SetWorkerCount(n int) (int, error) {
	if n < MinWorkers || n > MaxWorkers {
		return WorkerCount(), fmt.Errorf("workers must be %d–%d", MinWorkers, MaxWorkers)
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
	if n < MinWorkers || n > MaxWorkers {
		return workerCount, fmt.Errorf("workers must be %d–%d", MinWorkers, MaxWorkers)
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
