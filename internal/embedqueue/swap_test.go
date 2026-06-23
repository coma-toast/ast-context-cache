package embedqueue

import (
	"testing"
	"time"
)

type stubEmbedder struct{}

func (stubEmbedder) Embed(texts []string) ([][]float32, error) { return nil, nil }
func (stubEmbedder) EmbedSingle(text string) ([]float32, error) { return nil, nil }

func TestPrepareForEmbedderSwap_pausesWorkers(t *testing.T) {
	Start(stubEmbedder{})
	if _, err := SetWorkerCount(1); err != nil {
		t.Fatal(err)
	}
	PrepareForEmbedderSwap(5 * time.Second)
	if WorkerCount() != 0 {
		t.Fatalf("WorkerCount() = %d, want 0 during swap prep", WorkerCount())
	}
	RestoreWorkersAfterSwap()
	if WorkerCount() != 1 {
		t.Fatalf("WorkerCount() = %d, want 1 after restore", WorkerCount())
	}
	SetWorkerCount(0)
}

func TestEnqueuePendingRetry_nonBlocking(t *testing.T) {
	Start(stubEmbedder{})
	pendingCh = make(chan job, 1)
	pendingCh <- job{file: "fill", projectPath: "/p"}
	pendingMu.Lock()
	pending = map[string]job{"a\x00/p": {file: "a", projectPath: "/p"}}
	pendingChQueued = map[string]struct{}{}
	pendingMu.Unlock()
	done := make(chan struct{})
	go func() {
		enqueuePendingRetry(job{file: "b", projectPath: "/p"})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("enqueuePendingRetry blocked on full pendingCh")
	}
}
