package embedqueue

import (
	"testing"
	"time"
)

type stubEmbedder struct{}

func (stubEmbedder) Embed(texts []string) ([][]float32, error)  { return nil, nil }
func (stubEmbedder) EmbedSingle(text string) ([]float32, error) { return nil, nil }

type cancelingEmbedder struct {
	stubEmbedder
	canceled bool
}

func (c *cancelingEmbedder) CancelInFlight() {
	c.canceled = true
}

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

func TestPrepareForEmbedderSwap_nestedPause(t *testing.T) {
	Start(stubEmbedder{})
	if _, err := SetWorkerCount(3); err != nil {
		t.Fatal(err)
	}
	PrepareForEmbedderSwap(5 * time.Second)
	PrepareForEmbedderSwap(5 * time.Second)
	if WorkerCount() != 0 {
		t.Fatalf("WorkerCount() = %d, want 0 during nested swap prep", WorkerCount())
	}
	RestoreWorkersAfterSwap()
	if WorkerCount() != 0 {
		t.Fatalf("WorkerCount() = %d, want 0 after first restore (still nested)", WorkerCount())
	}
	RestoreWorkersAfterSwap()
	if WorkerCount() != 3 {
		t.Fatalf("WorkerCount() = %d, want 3 after full restore", WorkerCount())
	}
	SetWorkerCount(0)
}

// Uses its own full channel: pendingCh is shared with the workers earlier
// tests started, which read it while running — swapping it out raced them.
func TestEnqueuePendingRetry_nonBlocking(t *testing.T) {
	full := make(chan job, 1)
	full <- job{file: "fill", projectPath: "/p"}
	j := job{file: "b", projectPath: "/p"}
	k := jobKey(j)
	pendingMu.Lock()
	if pending == nil {
		pending = map[string]job{}
	}
	pending[k] = j // only pending jobs are retried
	pendingMu.Unlock()
	t.Cleanup(func() {
		pendingMu.Lock()
		delete(pending, k)
		delete(pendingChQueued, k)
		pendingMu.Unlock()
	})
	done := make(chan bool, 1)
	go func() { done <- enqueuePendingRetryOn(full, j) }()
	select {
	case queued := <-done:
		if queued {
			t.Fatal("enqueuePendingRetry reported queuing onto a full channel")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("enqueuePendingRetry blocked on full pendingCh")
	}
	pendingMu.Lock()
	_, marked := pendingChQueued[k]
	pendingMu.Unlock()
	if marked {
		t.Fatal("a retry that wasn't queued must not stay marked queued")
	}
}

func TestPrepareForEmbedderSwap_cancelsInFlightRemoteRequests(t *testing.T) {
	Start(stubEmbedder{})
	c := &cancelingEmbedder{}
	SetEmbedder(c)
	if _, err := SetWorkerCount(0); err != nil {
		t.Fatal(err)
	}
	PrepareForEmbedderSwap(10 * time.Millisecond)
	if !c.canceled {
		t.Fatal("expected swap prep to cancel in-flight embedder requests")
	}
	RestoreWorkersAfterSwap()
}
