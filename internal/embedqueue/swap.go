package embedqueue

import (
	"log"
	"sync/atomic"
	"time"
)

const defaultSwapDrainTimeout = 2 * time.Minute

var (
	swapRestoreWorkers int
	swapPauseDepth     int
)

// SwapPaused reports whether workers are paused for embedder swap or WAL maintenance.
func SwapPaused() bool {
	workerMu.Lock()
	defer workerMu.Unlock()
	return swapPauseDepth > 0
}

func cancelInFlightEmbedderRequests(e interface{}) {
	if c, ok := e.(interface{ CancelInFlight() }); ok {
		c.CancelInFlight()
	}
}

// PrepareForEmbedderSwap stops workers and waits for in-flight embeds before backend reload.
func PrepareForEmbedderSwap(timeout time.Duration) {
	if !workersStarted() {
		return
	}
	if timeout <= 0 {
		timeout = defaultSwapDrainTimeout
	}
	workerMu.Lock()
	swapPauseDepth++
	if swapPauseDepth == 1 {
		swapRestoreWorkers = workerCount
		if workerCount > 0 {
			if err := applyWorkerCountLocked(0, false); err != nil {
				log.Printf("embedqueue: swap prep pause workers: %v", err)
			} else {
				log.Printf("embedqueue: paused %d workers for embedder swap", swapRestoreWorkers)
			}
		}
	}
	workerMu.Unlock()
	cancelInFlightEmbedderRequests(queueEmbedder())
	deadline := time.Now().Add(timeout)
	for atomic.LoadInt64(&inFlight) > 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if n := atomic.LoadInt64(&inFlight); n > 0 {
		log.Printf("embedqueue: swap prep timed out with %d in-flight embed(s) still running", n)
	}
	if n := DrainQueueToPending(); n > 0 {
		log.Printf("embedqueue: drained %d queued jobs before embedder swap", n)
	}
}

// RestoreWorkersAfterSwap resumes workers paused by PrepareForEmbedderSwap (not persisted).
func RestoreWorkersAfterSwap() {
	if !workersStarted() {
		swapRestoreWorkers = 0
		swapPauseDepth = 0
		return
	}
	workerMu.Lock()
	defer workerMu.Unlock()
	if swapPauseDepth <= 0 {
		return
	}
	swapPauseDepth--
	if swapPauseDepth > 0 {
		return
	}
	n := swapRestoreWorkers
	swapRestoreWorkers = 0
	if n <= 0 {
		return
	}
	max := MaxWorkers()
	if n > max {
		n = max
	}
	if err := applyWorkerCountLocked(n, false); err != nil {
		log.Printf("embedqueue: restore workers after swap: %v", err)
		return
	}
	log.Printf("embedqueue: restored %d workers after embedder swap", n)
}
