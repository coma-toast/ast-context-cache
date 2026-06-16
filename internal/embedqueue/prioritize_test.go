package embedqueue

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

func TestEnsureProjectEmbeddings_drainsLowQueueForProject(t *testing.T) {
	highCh = make(chan job, 8)
	lowCh = make(chan job, 8)
	pendingCh = make(chan job, 8)
	pendingMu.Lock()
	pending = map[string]job{}
	pendingChQueued = map[string]struct{}{}
	pendingMu.Unlock()
	emb = &noopEmbedder{}
	embedder.MarkReady()

	proj := "/tmp/query-boost-proj"
	other := "/tmp/other-proj"
	lowCh <- job{file: "/tmp/a.go", projectPath: proj}
	lowCh <- job{file: "/tmp/b.go", projectPath: other}
	lowCh <- job{file: "/tmp/c.go", projectPath: proj}

	EnsureProjectEmbeddings(proj)

	if channelsContainProject(proj) {
		t.Fatal("project jobs should be drained")
	}
	if len(lowCh) != 1 {
		t.Fatalf("other project job should remain queued, low used=%d", len(lowCh))
	}
}

func TestPromotePendingToHigh(t *testing.T) {
	highCh = make(chan job, 4)
	lowCh = make(chan job, 4)
	pendingCh = make(chan job, 4)
	pendingMu.Lock()
	pending = map[string]job{
		jobKey(job{file: "/tmp/p.go", projectPath: "/proj"}): {file: "/tmp/p.go", projectPath: "/proj"},
	}
	pendingChQueued = map[string]struct{}{}
	pendingMu.Unlock()
	if n := promotePendingToHigh("/proj"); n != 1 {
		t.Fatalf("promoted=%d", n)
	}
	if PendingCount() != 0 {
		t.Fatalf("pending=%d", PendingCount())
	}
	select {
	case <-highCh:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected job on high channel")
	}
}
