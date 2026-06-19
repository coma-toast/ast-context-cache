package embedqueue

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func TestFlushPendingIfReadyQueuesJobs(t *testing.T) {
	pendingCh = make(chan job, 4)
	highCh = make(chan job, 4)
	lowCh = make(chan job, 4)
	emb = &noopEmbedder{}
	pendingMu.Lock()
	pending = map[string]job{jobKey(job{file: "/tmp/a.go", projectPath: "/proj"}): {file: "/tmp/a.go", projectPath: "/proj"}}
	pendingMu.Unlock()
	embedder.MarkReady()
	flushPendingIfReady()
	select {
	case <-pendingCh:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected pending job queued")
	}
}

func TestDrainQueueToPendingOnError(t *testing.T) {
	pendingCh = make(chan job, 4)
	highCh = make(chan job, 4)
	lowCh = make(chan job, 4)
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	highCh <- job{file: "/tmp/a.go", projectPath: "/proj"}
	lowCh <- job{file: "/tmp/b.go", projectPath: "/proj"}
	if n := DrainQueueToPending(); n != 2 {
		t.Fatalf("drained=%d", n)
	}
	if PendingCount() != 2 {
		t.Fatalf("pending=%d", PendingCount())
	}
	s := Snapshot()
	if s.Queued != 0 {
		t.Fatalf("queued=%d", s.Queued)
	}
}

func TestRecoveryBeforeSyncRace(t *testing.T) {
	pendingCh = make(chan job, 8)
	highCh = make(chan job, 8)
	lowCh = make(chan job, 8)
	emb = &noopEmbedder{}
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	embedder.MarkReady()
	markPendingIfNew(job{file: "/tmp/late.go", projectPath: "/proj"}, pendingReasonSync)
	flushPendingIfReady()
	if PendingCount() != 1 {
		t.Fatalf("pending=%d", PendingCount())
	}
	select {
	case j := <-pendingCh:
		if j.file != "/tmp/late.go" {
			t.Fatalf("queued %s", j.file)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected auto-flush after sync while ready")
	}
}

func TestPurgeOrphanCodeVectors(t *testing.T) {
	project := "orphan-test"
	db.DB.Exec(`DELETE FROM vectors WHERE project_path = ?`, project)
	db.DB.Exec(`DELETE FROM symbols WHERE project_path = ?`, project)
	res, err := db.DB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, embed_hash) VALUES ('f', 'function', '/x.go', 1, 1, '', 'x.f', ?, 'abc')`, project)
	if err != nil {
		t.Fatal(err)
	}
	symID, _ := res.LastInsertId()
	_, err = db.DB.Exec(`INSERT INTO vectors (symbol_id, content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (?, 'abc', ?, 'code', '/x.go', 'f', 'function', ?)`,
		symID, []byte{1, 2, 3}, project)
	if err != nil {
		t.Fatal(err)
	}
	db.DB.Exec(`DELETE FROM symbols WHERE id = ?`, symID)
	if n := search.PurgeOrphanCodeVectors(); n != 1 {
		t.Fatalf("purged=%d", n)
	}
}

func TestStaleVectorDetectedByEmbedHashSQL(t *testing.T) {
	project := "stale-hash-test"
	file := "/tmp/stale.go"
	db.DB.Exec(`DELETE FROM vectors WHERE project_path = ?`, project)
	db.DB.Exec(`DELETE FROM symbols WHERE project_path = ?`, project)
	res, err := db.DB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, embed_hash) VALUES ('f', 'function', ?, 1, 1, '', 'stale.f', ?, 'newhash')`, file, project)
	if err != nil {
		t.Fatal(err)
	}
	symID, _ := res.LastInsertId()
	_, err = db.DB.Exec(`INSERT INTO vectors (symbol_id, content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (?, 'oldhash', ?, 'code', ?, 'f', 'function', ?)`,
		symID, []byte{1}, file, project)
	if err != nil {
		t.Fatal(err)
	}
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	if n := SyncPendingFromDB(); n != 1 {
		t.Fatalf("sync added=%d want 1", n)
	}
}

func TestFlushPendingDBBatch(t *testing.T) {
	project := "/proj-batch-test"
	db.DB.Exec(`DELETE FROM embed_pending WHERE project_path = ?`, project)
	pendingMu.Lock()
	for k, j := range pending {
		if j.projectPath == project {
			delete(pending, k)
		}
	}
	pendingMu.Unlock()
	for i := 0; i < 5; i++ {
		markPendingIfNew(job{file: "/tmp/batch" + string(rune('a'+i)) + ".go", projectPath: project}, pendingReasonFailed)
	}
	FlushPendingDB()
	var count int
	if err := db.DB.QueryRow(`SELECT COUNT(*) FROM embed_pending WHERE project_path = ?`, project).Scan(&count); err != nil || count != 5 {
		t.Fatalf("batch count=%d err=%v", count, err)
	}
}

func TestEmbedPendingPersistAndLoad(t *testing.T) {
	db.DB.Exec(`DELETE FROM embed_pending`)
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	db.DB.Exec(`DELETE FROM embed_pending WHERE file = ?`, "/tmp/persist.go")
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	if !markPendingIfNew(job{file: "/tmp/persist.go", projectPath: "/proj"}, pendingReasonFailed) {
		t.Fatal("expected new pending")
	}
	FlushPendingDB()
	var count int
	if err := db.DB.QueryRow(`SELECT COUNT(*) FROM embed_pending WHERE file = ?`, "/tmp/persist.go").Scan(&count); err != nil || count != 1 {
		t.Fatalf("persist row count=%d err=%v", count, err)
	}
	clearPending(job{file: "/tmp/persist.go", projectPath: "/proj"})
	FlushPendingDB()
	if err := db.DB.QueryRow(`SELECT COUNT(*) FROM embed_pending WHERE file = ?`, "/tmp/persist.go").Scan(&count); err != nil || count != 0 {
		t.Fatalf("after clear count=%d err=%v", count, err)
	}
	db.DB.Exec(`INSERT OR REPLACE INTO embed_pending (file, project_path, reason, updated_at) VALUES (?, ?, ?, ?)`,
		"/tmp/load.go", "/proj", pendingReasonFailed, time.Now().Unix())
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	if n := LoadPendingFromDB(); n != 1 {
		t.Fatalf("loaded=%d", n)
	}
	if PendingCount() != 1 {
		t.Fatalf("pending=%d", PendingCount())
	}
}

type noopEmbedder struct{}

func (n *noopEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1}
	}
	return out, nil
}

func (n *noopEmbedder) EmbedSingle(text string) ([]float32, error) {
	return []float32{1}, nil
}
