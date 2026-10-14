package memory

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// TestPruneArchivedKeepsRowsWhileIndexQuiesced tests that PruneArchived respects
// the index read gate: it must fail (and keep rows/vectors) when the index is
// quiesced, then succeed after the gate is released.
func TestPruneArchivedKeepsRowsWhileIndexQuiesced(t *testing.T) {
	testMemoryDB(t)
	t.Cleanup(func() { db.SetIndexReadGateForTest(false) })
	res, err := Store(StoreInput{Kind: KindFact, Scope: ScopeGlobal, Subject: "user.editor", Predicate: "is", Object: "vim"})
	if err != nil {
		t.Fatal(err)
	}
	old := time.Now().AddDate(0, 0, -100).Format("2006-01-02 15:04:05")
	if _, err := db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ? WHERE ref = ?`, old, res.Ref); err != nil {
		t.Fatal(err)
	}
	key := memoryVectorKey(res.Ref)
	vec := make([]float32, search.VectorDims)
	vec[0] = 1
	if err := search.Cache.Upsert([]search.VectorEntry{{
		ContentHash: search.ContentHash(key), DocType: "memory", SourceFile: key, Name: key, ProjectPath: "mem-q", Vector: vec,
	}}); err != nil {
		t.Fatal(err)
	}

	db.SetIndexReadGateForTest(true)
	if _, err := PruneArchived(90); err == nil {
		t.Fatal("PruneArchived succeeded while the index was quiesced")
	}
	db.SetIndexReadGateForTest(false)

	// After the gate is released, pruning should succeed
	if n, err := PruneArchived(90); err != nil {
		t.Fatalf("PruneArchived after gate release: err=%v want nil", err)
	} else if n != 1 {
		t.Fatalf("PruneArchived after gate release: pruned=%d want 1", n)
	}
	assertGone(t, res.Ref)
	if n := memoryVectorRows(t, key); n != 0 {
		t.Fatalf("after prune: vector rows=%d want 0", n)
	}
	for _, e := range search.Cache.GetAll("mem-q") {
		if e.SourceFile == key {
			t.Fatal("pruned memory's vector is still in the search cache")
		}
	}
}

func memoryVectorRows(t *testing.T, key string) int {
	t.Helper()
	var n int
	if err := db.IndexDB.QueryRow(`SELECT COUNT(*) FROM vectors WHERE doc_type = 'memory' AND source_file = ?`, key).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}
