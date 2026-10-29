package embedqueue

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

func TestEnsureProjectEmbeddings_drainsLowQueueForProject(t *testing.T) {
	l := resetLanesForTest(t, 8)
	SetEmbedder(&noopEmbedder{})
	embedder.MarkReady()

	proj := "/tmp/query-boost-proj"
	other := "/tmp/other-proj"
	l.low <- job{file: "/tmp/a.go", projectPath: proj}
	l.low <- job{file: "/tmp/b.go", projectPath: other}
	l.low <- job{file: "/tmp/c.go", projectPath: proj}

	EnsureProjectEmbeddings(proj)

	if channelsContainProject(proj) {
		t.Fatal("project jobs should be drained")
	}
	if len(l.low) != 1 {
		t.Fatalf("other project job should remain queued, low used=%d", len(l.low))
	}
}

func TestPromotePendingToHigh(t *testing.T) {
	l := resetLanesForTest(t, 4)
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
	case <-l.high:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected job on high channel")
	}
}
