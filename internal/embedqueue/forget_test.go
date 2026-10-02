package embedqueue

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

type countingEmbedder struct{ calls atomic.Int64 }

func (c *countingEmbedder) Embed(texts []string) ([][]float32, error) {
	c.calls.Add(1)
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1}
	}
	return out, nil
}

func (c *countingEmbedder) EmbedSingle(string) ([]float32, error) {
	c.calls.Add(1)
	return []float32{1}, nil
}

func setupDeletedFileJob(t *testing.T) (job, string) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	prev := indexer.OnFilePurged
	indexer.OnFilePurged = ForgetFile
	t.Cleanup(func() { indexer.OnFilePurged = prev })
	embedder.MarkReady()
	proj := t.TempDir()
	file := filepath.Join(proj, "dashboard.go")
	if err := os.WriteFile(file, []byte("package p\n\nfunc Dashboard() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := indexer.IndexFile(file, proj); err != nil {
		t.Fatal(err)
	}
	j := job{file: file, projectPath: proj}
	t.Cleanup(func() {
		pendingMu.Lock()
		delete(pending, jobKey(j))
		pendingMu.Unlock()
	})
	if !markPendingIfNew(j, pendingReasonFailed) {
		t.Fatal("expected job to become pending")
	}
	FlushPendingDB()
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	return j, proj
}

func isPending(j job) bool {
	pendingMu.Lock()
	defer pendingMu.Unlock()
	_, ok := pending[jobKey(j)]
	return ok
}

func embedPendingRows(t *testing.T, j job) int {
	t.Helper()
	FlushPendingDB()
	var n int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM embed_pending WHERE file = ? AND project_path = ?`, j.file, j.projectPath).Scan(&n)
	return n
}

// A pending retry for a file deleted from disk must be dropped and the file's
// rows purged, not re-embedded ("Embedded 13 symbols from …/dashboard.py").
func TestRunDropsPendingJobForDeletedFile(t *testing.T) {
	j, proj := setupDeletedFileJob(t)
	emb := &countingEmbedder{}
	runWithEmbedder(j, emb, false)
	if n := emb.calls.Load(); n != 0 {
		t.Fatalf("embedder called %d times for a deleted file", n)
	}
	if isPending(j) {
		t.Fatal("deleted file still pending retry")
	}
	if n := embedPendingRows(t, j); n != 0 {
		t.Fatalf("embed_pending rows=%d want 0", n)
	}
	var syms int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`, j.file, proj).Scan(&syms)
	if syms != 0 {
		t.Fatalf("symbols=%d want 0 after purge", syms)
	}
	// Recovery must not resurrect it from the DB either.
	if n := syncPendingFromDBLocked(); n != 0 || isPending(j) {
		t.Fatalf("recovery re-queued %d files (deleted file pending=%v)", n, isPending(j))
	}
}

func TestForgetFileOnPurgeClearsPendingRetry(t *testing.T) {
	j, _ := setupDeletedFileJob(t)
	if err := indexer.PurgeFile(j.file, j.projectPath); err != nil {
		t.Fatal(err)
	}
	if isPending(j) {
		t.Fatal("purged file still pending retry")
	}
	if n := embedPendingRows(t, j); n != 0 {
		t.Fatalf("embed_pending rows=%d want 0", n)
	}
}
