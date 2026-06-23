package embedqueue

import (
	"fmt"
	"log"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	defaultAuxWorkers        = 0
	DefaultAuxMaxWorkers     = 10
	absoluteAuxMaxWorkers    = 32
	embedAuxWorkersSetting   = "EMBED_AUX_WORKERS"
	embedAuxWorkerMaxSetting = "embed_aux_worker_max"
)

var (
	auxEmb          embedder.Interface
	auxEmbMu        sync.RWMutex
	auxWorkerCount  = defaultAuxWorkers
	auxWorkerTarget = defaultAuxWorkers
	auxWorkerLive   atomic.Int64
	auxWorkerMu     sync.Mutex
	auxWorkerStop   chan struct{}
	auxStarted      sync.Once
)

// SetAuxEmbedder sets the catch-up embedder for aux workers.
func SetAuxEmbedder(e embedder.Interface) {
	auxEmbMu.Lock()
	auxEmb = e
	auxEmbMu.Unlock()
}

func queueAuxEmbedder() embedder.Interface {
	auxEmbMu.RLock()
	defer auxEmbMu.RUnlock()
	return auxEmb
}

// AuxMaxWorkers returns the upper limit for aux embed workers.
func AuxMaxWorkers() int {
	raw := db.GetSetting(embedAuxWorkerMaxSetting, strconv.Itoa(DefaultAuxMaxWorkers))
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return DefaultAuxMaxWorkers
	}
	if n > absoluteAuxMaxWorkers {
		return absoluteAuxMaxWorkers
	}
	return n
}

func loadAuxWorkerCount() int {
	raw := db.GetSetting(embedAuxWorkersSetting, strconv.Itoa(defaultAuxWorkers))
	n, err := strconv.Atoi(raw)
	if err != nil || n < MinWorkers {
		return defaultAuxWorkers
	}
	max := AuxMaxWorkers()
	if n > max {
		return max
	}
	return n
}

func persistAuxWorkerCount(n int) {
	if err := db.SetSetting(embedAuxWorkersSetting, strconv.Itoa(n)); err != nil {
		log.Printf("embedqueue: persist aux workers: %v", err)
	}
}

// AuxWorkerCount returns configured aux worker goroutines.
func AuxWorkerCount() int {
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	return auxWorkerCount
}

// AuxWorkerLive returns running aux worker goroutines.
func AuxWorkerLive() int {
	return int(auxWorkerLive.Load())
}

func applyAuxWorkerCountLocked(n int, persist bool) error {
	if auxWorkerStop == nil {
		return fmt.Errorf("aux embed queue not started")
	}
	for auxWorkerCount < n {
		go auxWorker()
		auxWorkerCount++
	}
	for auxWorkerCount > n {
		auxWorkerStop <- struct{}{}
		auxWorkerCount--
	}
	if persist {
		auxWorkerTarget = auxWorkerCount
		persistAuxWorkerCount(auxWorkerCount)
		log.Printf("embed queue: aux workers set to %d", auxWorkerCount)
	}
	realtime.Notify(realtime.EmbedFinished | realtime.IndexHealth)
	return nil
}

// SetAuxWorkerCount changes aux worker pool size.
func SetAuxWorkerCount(n int) (int, error) {
	max := AuxMaxWorkers()
	if n < MinWorkers || n > max {
		return AuxWorkerCount(), fmt.Errorf("aux workers must be %d–%d", MinWorkers, max)
	}
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	if err := applyAuxWorkerCountLocked(n, true); err != nil {
		return auxWorkerCount, err
	}
	return auxWorkerCount, nil
}

// AdjustAuxWorkers atomically adds delta to aux worker count.
func AdjustAuxWorkers(delta int) (int, error) {
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	n := auxWorkerCount + delta
	max := AuxMaxWorkers()
	if n < MinWorkers || n > max {
		return auxWorkerCount, fmt.Errorf("aux workers must be %d–%d", MinWorkers, max)
	}
	if err := applyAuxWorkerCountLocked(n, true); err != nil {
		return auxWorkerCount, err
	}
	return auxWorkerCount, nil
}

// ClampAuxWorkersToMax lowers aux worker count when the configured max shrinks.
func ClampAuxWorkersToMax() error {
	auxWorkerMu.Lock()
	defer auxWorkerMu.Unlock()
	max := AuxMaxWorkers()
	if auxWorkerCount <= max {
		return nil
	}
	return applyAuxWorkerCountLocked(max, true)
}

// StartAux launches aux worker goroutines sharing the primary job channels.
func StartAux(e embedder.Interface) {
	if e == nil {
		return
	}
	auxStarted.Do(func() {
		SetAuxEmbedder(e)
		auxWorkerStop = make(chan struct{}, absoluteAuxMaxWorkers)
		auxWorkerMu.Lock()
		auxWorkerCount = loadAuxWorkerCount()
		auxWorkerTarget = auxWorkerCount
		n := auxWorkerCount
		auxWorkerMu.Unlock()
		for i := 0; i < n; i++ {
			go auxWorker()
		}
		log.Printf("embed queue: %d aux workers (backend catch-up pool)", n)
	})
}

func auxWorker() {
	auxWorkerLive.Add(1)
	defer func() {
		auxWorkerLive.Add(-1)
		notifyWorkerPoolLiveChange()
	}()
	waitForProcessingReady()
	for {
		select {
		case <-auxWorkerStop:
			return
		default:
		}
		select {
		case <-auxWorkerStop:
			return
		case j := <-pendingCh:
			runWithEmbedder(j, queueAuxEmbedder())
		default:
			select {
			case <-auxWorkerStop:
				return
			case j := <-pendingCh:
				runWithEmbedder(j, queueAuxEmbedder())
			case j := <-highCh:
				runWithEmbedder(j, queueAuxEmbedder())
			case j := <-lowCh:
				runWithEmbedder(j, queueAuxEmbedder())
			}
		}
	}
}
