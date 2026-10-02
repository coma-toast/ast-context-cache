package contextnotes

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// A WAL quiesce gates index writes. Deleting a note while it is up must fail with
// the note kept, not delete the note and silently skip its vector, which then
// reloads on restart and stays searchable.
func TestFlushFailsAndKeepsNoteWhileIndexQuiesced(t *testing.T) {
	testNotesDB(t)
	t.Cleanup(func() { db.SetIndexReadGateForTest(false) })
	res, err := Store("q-sess", "note body", "lbl", "", nil, "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	seedNoteVector(t, res.Ref, "q-sess")

	db.SetIndexReadGateForTest(true)
	if _, err := Flush("q-sess", []string{res.Ref}, "", false); err == nil {
		t.Fatal("flush succeeded while the index was quiesced")
	}
	db.SetIndexReadGateForTest(false)
	if !noteExists(t, res.Ref) || noteVectorRows(t, res.Ref) != 1 {
		t.Fatalf("after failed flush: note exists=%v vector rows=%d, want both kept", noteExists(t, res.Ref), noteVectorRows(t, res.Ref))
	}

	flushed, err := Flush("q-sess", []string{res.Ref}, "", false)
	if err != nil {
		t.Fatal(err)
	}
	if flushed.FlushedRefs != 1 || noteExists(t, res.Ref) || noteVectorRows(t, res.Ref) != 0 || noteVectorCached(res.Ref, "q-sess") {
		t.Fatalf("after retry: flushed=%d note exists=%v vector rows=%d cached=%v, want all gone",
			flushed.FlushedRefs, noteExists(t, res.Ref), noteVectorRows(t, res.Ref), noteVectorCached(res.Ref, "q-sess"))
	}
}

// LRU eviction deletes the session's oldest note and checks the limit again. If
// that delete fails, the same note is still the oldest, so eviction must give up
// rather than loop forever.
func TestLRUEvictionFailsWhileIndexQuiesced(t *testing.T) {
	testNotesDB(t)
	t.Cleanup(func() { db.SetIndexReadGateForTest(false) })
	db.SetSetting("context_max_notes_session", "1")
	db.SetSetting("context_limit_policy", "lru_session")
	r1, err := Store("lru-q", "one", "1", "", nil, "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	seedNoteVector(t, r1.Ref, "lru-q")

	db.SetIndexReadGateForTest(true)
	done := make(chan error, 1)
	go func() {
		_, err := Store("lru-q", "two", "2", "", nil, "", nil, nil)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("store evicted a note while the index was quiesced")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("eviction never returned")
	}
	db.SetIndexReadGateForTest(false)
	if !noteExists(t, r1.Ref) || noteVectorRows(t, r1.Ref) != 1 {
		t.Fatal("failed eviction deleted the oldest note or its vector")
	}
}

func seedNoteVector(t *testing.T, ref, sessionID string) {
	t.Helper()
	vec := make([]float32, search.VectorDims)
	vec[0] = 1
	if err := search.Cache.Upsert([]search.VectorEntry{{
		ContentHash: search.ContentHash(noteVectorKey(ref)),
		DocType:     "note",
		SourceFile:  noteVectorKey(ref),
		Name:        ref,
		ProjectPath: sessionID,
		Vector:      vec,
	}}); err != nil {
		t.Fatal(err)
	}
}

func noteExists(t *testing.T, ref string) bool {
	t.Helper()
	var n int
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_notes WHERE ref = ?`, ref).Scan(&n)
	return n == 1
}

func noteVectorRows(t *testing.T, ref string) int {
	t.Helper()
	var n int
	if err := db.IndexDB.QueryRow(`SELECT COUNT(*) FROM vectors WHERE doc_type = 'note' AND source_file = ?`, noteVectorKey(ref)).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func noteVectorCached(ref, sessionID string) bool {
	for _, e := range search.Cache.GetAll(sessionID) {
		if e.DocType == "note" && e.SourceFile == noteVectorKey(ref) {
			return true
		}
	}
	return false
}
