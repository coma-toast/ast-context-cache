package purge

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// A WAL quiesce can land while the index data is being purged, so index writes
// are gated by the time the project's notes and memory are deleted from
// context.db. Those rows are deleted regardless, so their vectors must already be
// gone by then. Otherwise they are orphaned in index.db, reload on the next
// restart, and stay searchable for a project that no longer exists.
func TestProjectDataLeavesNoNoteOrMemoryVectorsWhenQuiescedMidPurge(t *testing.T) {
	home := dbtest.Init(t)
	t.Cleanup(func() {
		afterSymbolDelete = nil
		db.SetIndexReadGateForTest(false)
	})
	db.WaitInitFTSRebuildForTest()

	gone := filepath.Join(home, "git", "gone")
	kept := filepath.Join(home, "git", "kept")
	for _, p := range []string{gone, kept} {
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatal(err)
		}
		seedProject(t, p)
		seedRefVectors(t, p)
	}

	quiesced := make(chan error, 1)
	afterSymbolDelete = func() error {
		go func() { quiesced <- db.QuiesceIndexPoolForTest() }()
		// The quiesce waits for this transaction to commit before closing the pool,
		// so carry on once it has gated index reads and writes.
		deadline := time.Now().Add(5 * time.Second)
		for !db.IndexReadQuiesced() {
			if time.Now().After(deadline) {
				return errors.New("quiesce never gated index reads")
			}
			time.Sleep(time.Millisecond)
		}
		return nil
	}
	if err := ProjectData(gone); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-quiesced:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("quiesce never finished")
	}
	if err := db.RestoreIndexPoolForTest(); err != nil {
		t.Fatal(err)
	}

	if n := refVectorRows(t, gone); n != 0 {
		t.Fatalf("%d note/memory vector rows left in index.db for the purged project", n)
	}
	if got := cachedRefVectors(gone); len(got) != 0 {
		t.Fatalf("note/memory vectors still cached for the purged project: %v", got)
	}
	assertPurged(t, gone)

	if n := refVectorRows(t, kept); n != 2 {
		t.Fatalf("other project: note/memory vector rows=%d want 2", n)
	}
	if got := cachedRefVectors(kept); len(got) != 2 {
		t.Fatalf("other project: cached note/memory vectors=%v want 2", got)
	}
}

// If the note/memory vectors can't be deleted, the purge must fail before it
// deletes anything, leaving the project indexed so the deleted-project sweep
// finds it again and finishes the job.
func TestProjectDataFailsWhenQuiescedAndSweepRetries(t *testing.T) {
	home := dbtest.Init(t)
	t.Cleanup(func() { db.SetIndexReadGateForTest(false) })

	gone := filepath.Join(home, "git", "gone")
	if err := os.MkdirAll(gone, 0755); err != nil {
		t.Fatal(err)
	}
	seedProject(t, gone)
	seedRefVectors(t, gone)

	db.SetIndexReadGateForTest(true)
	err := ProjectData(gone)
	db.SetIndexReadGateForTest(false)
	if err == nil {
		t.Fatal("purge succeeded while the index was quiesced")
	}
	if n := refVectorRows(t, gone); n != 2 {
		t.Fatalf("after failed purge: note/memory vector rows=%d want 2", n)
	}
	var notes, memories int
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_notes WHERE project_path = ?`, gone).Scan(&notes)
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE project_path = ?`, gone).Scan(&memories)
	if notes != 1 || memories != 1 {
		t.Fatalf("after failed purge: notes=%d memories=%d want 1 each", notes, memories)
	}
	if symbolCount(t, gone) == 0 {
		t.Fatal("failed purge deleted the symbols, so the sweep can no longer find the project")
	}

	if err := os.RemoveAll(gone); err != nil {
		t.Fatal(err)
	}
	if purged := SweepDeletedProjectsNow(); !slices.Contains(purged, gone) {
		t.Fatalf("sweep purged %v, want it to retry %s", purged, gone)
	}
	if n := refVectorRows(t, gone); n != 0 {
		t.Fatalf("after sweep: %d note/memory vector rows left", n)
	}
	assertPurged(t, gone)
}

// seedRefVectors stores vectors for the note and memory seedProject creates,
// keyed the way contextnotes.EmbedNote and memory.EmbedEntry key them.
func seedRefVectors(t *testing.T, projectPath string) {
	t.Helper()
	vec := make([]float32, search.VectorDims)
	vec[0] = 1
	var entries []search.VectorEntry
	for docType, key := range refVectorKeys(projectPath) {
		entries = append(entries, search.VectorEntry{
			ContentHash: search.ContentHash(key),
			DocType:     docType,
			SourceFile:  key,
			Name:        key,
			ProjectPath: "sess",
			Vector:      vec,
		})
	}
	if err := search.Cache.Upsert(entries); err != nil {
		t.Fatal(err)
	}
}

func refVectorKeys(projectPath string) map[string]string {
	return map[string]string{"note": "note:note-" + projectPath, "memory": "mem:mem-" + projectPath}
}

func refVectorRows(t *testing.T, projectPath string) int {
	t.Helper()
	keys := refVectorKeys(projectPath)
	var n int
	if err := db.IndexDB.QueryRow(`SELECT COUNT(*) FROM vectors WHERE (doc_type = 'note' AND source_file = ?) OR (doc_type = 'memory' AND source_file = ?)`,
		keys["note"], keys["memory"]).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func cachedRefVectors(projectPath string) []string {
	keys := refVectorKeys(projectPath)
	var out []string
	for _, e := range search.Cache.GetAll("sess") {
		if keys[e.DocType] == e.SourceFile {
			out = append(out, e.SourceFile)
		}
	}
	return out
}
