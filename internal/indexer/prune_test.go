package indexer

import (
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

// testPruneDB gives the test its own empty database. dbtest closes it (stopping
// the index writer, batchers and the startup FTS rebuild) before its TempDir is
// removed; the pools used to be left open, and the rebuild writing into .astcache
// during RemoveAll failed cleanup with "directory not empty". The package's
// TestMain database is closed first and reopened afterwards (cleanups run last-in
// first-out, so HOME is restored by then) for tests that rely on it.
func testPruneDB(t *testing.T) {
	t.Helper()
	db.Close()
	t.Cleanup(func() {
		if err := db.Init(); err != nil {
			t.Fatal(err)
		}
	})
	dbtest.Init(t)
}

func writeGo(t *testing.T, path, fn string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package p\n\nimport \"fmt\"\n\nfunc " + fn + "() { fmt.Println(1) }\n"
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
}

func countRows(t *testing.T, query string, args ...interface{}) int {
	t.Helper()
	var n int
	if err := db.IndexDB.QueryRow(query, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// fileRowCounts counts every per-file row PurgeFile is responsible for.
func fileRowCounts(t *testing.T, file, project string) map[string]int {
	t.Helper()
	return map[string]int{
		"symbols":       countRows(t, `SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`, file, project),
		"edges":         countRows(t, `SELECT COUNT(*) FROM edges WHERE source_file = ? AND project_path = ?`, file, project),
		"vectors":       countRows(t, `SELECT COUNT(*) FROM vectors WHERE source_file = ? AND project_path = ?`, file, project),
		"summaries":     countRows(t, `SELECT COUNT(*) FROM summaries WHERE file_path = ? AND project_path = ?`, file, project),
		"indexed_files": countRows(t, `SELECT COUNT(*) FROM indexed_files WHERE file = ? AND project_path = ?`, file, project),
		"embed_pending": countRows(t, `SELECT COUNT(*) FROM embed_pending WHERE file = ? AND project_path = ?`, file, project),
	}
}

func assertNoFileRows(t *testing.T, file, project string) {
	t.Helper()
	for table, n := range fileRowCounts(t, file, project) {
		if n != 0 {
			t.Errorf("%s: %d rows left for %s", table, n, file)
		}
	}
}

// seedAuxRows adds the rows that only exist once a file has been embedded /
// summarized / queued for retry, so purges are checked against all of them.
func seedAuxRows(t *testing.T, file, project string) {
	t.Helper()
	stmts := []struct {
		q    string
		args []interface{}
	}{
		{
			`INSERT INTO vectors (symbol_id, content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (0, ?, x'00', 'code', ?, 'X', 'function', ?)`,
			[]interface{}{"h-" + file, file, project},
		},
		{
			`INSERT INTO summaries (symbol_name, file_path, project_path, summary_text, content_hash) VALUES ('X', ?, ?, 's', 'h')`,
			[]interface{}{file, project},
		},
		{
			`INSERT OR REPLACE INTO embed_pending (file, project_path, reason, updated_at) VALUES (?, ?, 'failed', 0)`,
			[]interface{}{file, project},
		},
	}
	for _, s := range stmts {
		if _, err := db.IndexDB.Exec(s.q, s.args...); err != nil {
			t.Fatalf("seed %s: %v", s.q, err)
		}
	}
}

func TestIndexDirectoryPrunesFilesDeletedWhileDown(t *testing.T) {
	testPruneDB(t)
	proj := t.TempDir()
	keep := filepath.Join(proj, "keep.go")
	gone := filepath.Join(proj, "sub", "gone.go")
	writeGo(t, keep, "Keep")
	writeGo(t, gone, "Gone")
	if _, err := IndexDirectory(proj, proj); err != nil {
		t.Fatal(err)
	}
	seedAuxRows(t, gone, proj)
	// Rows for a file that never had an indexed_files entry (older index, partial
	// delete): indexed_files-only reconciliation never saw these.
	orphan := filepath.Join(proj, "orphan.go")
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('Orphan', 'function', ?, 1, 1, ?)`, orphan, proj); err != nil {
		t.Fatal(err)
	}
	// Deleted while no watcher was running.
	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	if _, err := IndexDirectory(proj, proj); err != nil {
		t.Fatal(err)
	}
	assertNoFileRows(t, gone, proj)
	assertNoFileRows(t, orphan, proj)
	if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`, keep, proj); n == 0 {
		t.Fatal("existing file lost its symbols")
	}
}

func TestPruneMissingFilesSkipsMissingRoot(t *testing.T) {
	testPruneDB(t)
	proj := filepath.Join(t.TempDir(), "unmounted")
	file := filepath.Join(proj, "a.go")
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('A', 'function', ?, 1, 1, ?)`, file, proj); err != nil {
		t.Fatal(err)
	}
	if n := PruneMissingFiles(proj, proj); n != 0 {
		t.Fatalf("pruned %d files with the project root missing", n)
	}
	if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE file = ?`, file); n != 1 {
		t.Fatalf("symbols=%d want 1", n)
	}
}

type countingEmbedder struct{ calls atomic.Int64 }

func (c *countingEmbedder) Embed(texts []string) ([][]float32, error) {
	c.calls.Add(1)
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1, 0}
	}
	return out, nil
}

func (c *countingEmbedder) EmbedSingle(string) ([]float32, error) {
	c.calls.Add(1)
	return []float32{1, 0}, nil
}

func TestEmbedFileSymbolsPurgesDeletedFileInsteadOfEmbedding(t *testing.T) {
	testPruneDB(t)
	proj := t.TempDir()
	file := filepath.Join(proj, "dashboard.go")
	writeGo(t, file, "Dashboard")
	if _, _, _, err := IndexFile(file, proj); err != nil {
		t.Fatal(err)
	}
	seedAuxRows(t, file, proj)
	var purged []string
	OnFilePurged = func(f, _ string) { purged = append(purged, f) }
	t.Cleanup(func() { OnFilePurged = nil })
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	emb := &countingEmbedder{}
	if err := EmbedFileSymbols(emb, file, proj); err != nil {
		t.Fatalf("EmbedFileSymbols: %v", err)
	}
	if n := emb.calls.Load(); n != 0 {
		t.Fatalf("embedder called %d times for a deleted file", n)
	}
	assertNoFileRows(t, file, proj)
	if len(purged) != 1 || purged[0] != file {
		t.Fatalf("OnFilePurged calls = %v, want [%s]", purged, file)
	}
}

func TestIndexDirectoryIndexesSymlinkedFileOnce(t *testing.T) {
	testPruneDB(t)
	proj := t.TempDir()
	real := filepath.Join(proj, "litellm_sync.go")
	link := filepath.Join(proj, "litellm-sync.go")
	writeGo(t, real, "LoadConfig")
	if err := os.Symlink("litellm_sync.go", link); err != nil {
		t.Fatal(err)
	}
	// Rows left by an earlier version that indexed the link as its own file.
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('LoadConfig', 'function', ?, 5, 5, ?)`, link, proj); err != nil {
		t.Fatal(err)
	}
	db.UpsertIndexedFile(link, proj, time.Now())
	seedAuxRows(t, link, proj)

	if _, err := IndexDirectory(proj, proj); err != nil {
		t.Fatal(err)
	}
	if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE name = 'LoadConfig' AND project_path = ?`, proj); n != 1 {
		t.Fatalf("LoadConfig indexed %d times, want 1", n)
	}
	if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`, real, proj); n != 1 {
		t.Fatalf("symbol not indexed under the real path (%d rows)", n)
	}
	assertNoFileRows(t, link, proj)

	// Indexing the link directly (index_files on one file, watcher event) is refused.
	if _, _, _, err := IndexFile(link, proj); !errors.Is(err, ErrSymlinkAlias) {
		t.Fatalf("IndexFile(link) err = %v, want ErrSymlinkAlias", err)
	}
	if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE name = 'LoadConfig' AND project_path = ?`, proj); n != 1 {
		t.Fatalf("after IndexFile(link): LoadConfig indexed %d times, want 1", n)
	}
}

func TestSymlinkOutsideProjectLoopsAndDangling(t *testing.T) {
	testPruneDB(t)
	base := t.TempDir()
	proj := filepath.Join(base, "proj")
	outside := filepath.Join(base, "elsewhere", "shared.go")
	writeGo(t, outside, "Shared")
	writeGo(t, filepath.Join(proj, "main.go"), "Main")
	outLink := filepath.Join(proj, "shared.go")
	loopA := filepath.Join(proj, "loop_a.go")
	loopB := filepath.Join(proj, "loop_b.go")
	dangling := filepath.Join(proj, "dangling.go")
	fifoLink := filepath.Join(proj, "fifo.go")
	// Reading a FIFO blocks forever; a link to one must never be opened.
	if err := syscall.Mkfifo(filepath.Join(base, "pipe"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, l := range []struct{ target, link string }{
		{outside, outLink},
		{"loop_b.go", loopA},
		{"loop_a.go", loopB},
		{"does_not_exist.go", dangling},
		{filepath.Join(base, "pipe"), fifoLink},
	} {
		if err := os.Symlink(l.target, l.link); err != nil {
			t.Fatal(err)
		}
	}

	if _, alias := SymlinkAlias(outLink, proj); alias {
		t.Fatal("symlink to a file outside the project must be indexed under the link path")
	}
	for _, p := range []string{loopA, loopB, dangling, fifoLink} {
		if _, alias := SymlinkAlias(p, proj); !alias {
			t.Fatalf("%s: expected skip (loop/dangling/non-regular)", p)
		}
	}

	done := make(chan error, 1)
	go func() { _, err := IndexDirectory(proj, proj); done <- err }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("IndexDirectory hung on symlinks")
	}
	if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE name = 'Shared' AND file = ? AND project_path = ?`, outLink, proj); n != 1 {
		t.Fatalf("outside-project link: %d Shared rows under link path, want 1", n)
	}
	for _, p := range []string{loopA, loopB, dangling, fifoLink} {
		if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE file = ?`, p); n != 0 {
			t.Fatalf("%s: %d symbols, want 0", p, n)
		}
	}
}

func TestSymlinkAliasTargetNotWalkedIsIndexedViaLink(t *testing.T) {
	testPruneDB(t)
	proj := t.TempDir()
	// Target lives in a skipped directory, so the link is the only indexed copy.
	target := filepath.Join(proj, "vendor", "lib.go")
	writeGo(t, target, "Lib")
	link := filepath.Join(proj, "lib.go")
	if err := os.Symlink(filepath.Join("vendor", "lib.go"), link); err != nil {
		t.Fatal(err)
	}
	if _, alias := SymlinkAlias(link, proj); alias {
		t.Fatal("link to a file the walk skips must not be treated as an alias")
	}
	// Symlinked directory inside the project: files reached through it alias the real ones.
	writeGo(t, filepath.Join(proj, "src", "x.go"), "X")
	if err := os.Symlink("src", filepath.Join(proj, "srclink")); err != nil {
		t.Fatal(err)
	}
	got, alias := SymlinkAlias(filepath.Join(proj, "srclink", "x.go"), proj)
	if !alias || got != filepath.Join(proj, "src", "x.go") {
		t.Fatalf("via symlinked dir: target=%q alias=%v", got, alias)
	}
}
