package docs

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// Doc section vectors live in index.db. Removing a source while a WAL quiesce
// gates index writes must fail with the source and its sections kept, rather than
// delete them and leave the vectors orphaned and searchable.
func TestRemoveSourceFailsAndKeepsSourceWhileIndexQuiesced(t *testing.T) {
	dbtest.Init(t)
	t.Cleanup(func() { db.SetIndexReadGateForTest(false) })
	id, err := AddSource("quiesce-docs", "markdown", "file:///nowhere.md", "")
	if err != nil {
		t.Fatal(err)
	}
	if err := storeEntries(id, []DocEntry{{Title: "Intro", Content: "hello"}}); err != nil {
		t.Fatal(err)
	}
	entries, err := ListEntriesBySource(id)
	if err != nil || len(entries) != 1 {
		t.Fatalf("entries=%v err=%v", entries, err)
	}
	key := docVectorKey(id, entries[0].ID)
	vec := make([]float32, search.VectorDims)
	vec[0] = 1
	if err := search.Cache.Upsert([]search.VectorEntry{{ContentHash: search.ContentHash(key), DocType: "doc", SourceFile: key, Name: "Intro", Vector: vec}}); err != nil {
		t.Fatal(err)
	}

	db.SetIndexReadGateForTest(true)
	if err := RemoveSource(id); err == nil {
		t.Fatal("removed a doc source while the index was quiesced")
	}
	db.SetIndexReadGateForTest(false)
	if got, _ := ListEntriesBySource(id); len(got) != 1 || docVectorRows(t, key) != 1 {
		t.Fatalf("after failed remove: entries=%d vector rows=%d, want 1 each", len(got), docVectorRows(t, key))
	}

	if err := RemoveSource(id); err != nil {
		t.Fatal(err)
	}
	if got, _ := ListEntriesBySource(id); len(got) != 0 || docVectorRows(t, key) != 0 {
		t.Fatalf("after remove: entries=%d vector rows=%d, want 0 each", len(got), docVectorRows(t, key))
	}
}

func docVectorRows(t *testing.T, key string) int {
	t.Helper()
	var n int
	if err := db.IndexDB.QueryRow(`SELECT COUNT(*) FROM vectors WHERE doc_type = 'doc' AND source_file = ?`, key).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}
