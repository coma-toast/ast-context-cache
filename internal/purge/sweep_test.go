package purge

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// present reports paths as existing unless listed in missing.
func present(missing ...string) func(string) bool {
	gone := map[string]bool{}
	for _, m := range missing {
		gone[m] = true
	}
	return func(p string) bool { return !gone[p] }
}

func TestRecordScanDebouncesBeforePurging(t *testing.T) {
	ResetMissCounts()
	paths := []string{"/a", "/b"}
	exists := present("/b")

	for i := 1; i < missesBeforePurge; i++ {
		if due := recordScan(paths, exists); len(due) != 0 {
			t.Fatalf("sweep %d purged too early: %v", i, due)
		}
		if got := MissCount("/b"); got != i {
			t.Fatalf("after sweep %d miss count=%d want %d", i, got, i)
		}
	}
	due := recordScan(paths, exists)
	if len(due) != 1 || due[0] != "/b" {
		t.Fatalf("due=%v want [/b] on sweep %d", due, missesBeforePurge)
	}
	// Counter is cleared after the purge so a recreated path starts fresh.
	if got := MissCount("/b"); got != 0 {
		t.Fatalf("miss count after purge=%d want 0", got)
	}
}

func TestRecordScanResetsWhenPathReturns(t *testing.T) {
	ResetMissCounts()
	paths := []string{"/a"}

	recordScan(paths, present("/a"))
	recordScan(paths, present("/a"))
	if got := MissCount("/a"); got != 2 {
		t.Fatalf("miss count=%d want 2", got)
	}
	// A transient blip (unmounted volume, wtg mid-operation) must not accumulate.
	if due := recordScan(paths, present()); len(due) != 0 {
		t.Fatalf("due=%v want none once path returned", due)
	}
	if got := MissCount("/a"); got != 0 {
		t.Fatalf("miss count after return=%d want 0", got)
	}
	if due := recordScan(paths, present("/a")); len(due) != 0 {
		t.Fatalf("due=%v want none: debounce should restart", due)
	}
}

func TestRecordScanForgetsUntrackedPaths(t *testing.T) {
	ResetMissCounts()
	recordScan([]string{"/a"}, present("/a"))
	if got := MissCount("/a"); got != 1 {
		t.Fatalf("miss count=%d want 1", got)
	}
	recordScan([]string{"/other"}, present("/other"))
	if got := MissCount("/a"); got != 0 {
		t.Fatalf("untracked path should be forgotten, count=%d", got)
	}
}

// Deleting a whole WTG space must take every repo under it, since each checkout
// is its own project_path.
func TestSweepPurgesEveryRepoOfADeletedSpace(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	ResetMissCounts()

	space := filepath.Join(home, "spaces", "throwaway")
	kept := filepath.Join(home, "spaces", "keeper", "slapi")
	repos := []string{
		filepath.Join(space, "slapi"),
		filepath.Join(space, "console"),
		filepath.Join(space, "sandbox"),
	}
	for _, p := range append(append([]string{}, repos...), kept) {
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatal(err)
		}
		seedProject(t, p)
	}

	// The space is deleted from disk; the unrelated space stays.
	if err := os.RemoveAll(space); err != nil {
		t.Fatal(err)
	}

	all := append(append([]string{}, repos...), kept)
	for i := 1; i < missesBeforePurge; i++ {
		if due := recordScan(all, dirExists); len(due) != 0 {
			t.Fatalf("purged before debounce elapsed: %v", due)
		}
		for _, p := range repos {
			if symbolCount(t, p) == 0 {
				t.Fatalf("%s purged too early", p)
			}
		}
	}

	due := recordScan(all, dirExists)
	if len(due) != len(repos) {
		t.Fatalf("due=%v want all %d repos of the deleted space", due, len(repos))
	}
	for _, p := range due {
		if err := ProjectData(p); err != nil {
			t.Fatalf("ProjectData(%s): %v", p, err)
		}
	}

	for _, p := range repos {
		assertPurged(t, p)
	}
	// The surviving space is untouched.
	if symbolCount(t, kept) == 0 {
		t.Fatalf("%s must not be purged", kept)
	}
}

// Spaces are created and destroyed frequently, so a user-initiated cleanup (Prune)
// must not make them wait out SweepDeletedProjects' consecutive-miss debounce.
func TestSweepDeletedProjectsNowPurgesImmediately(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	ResetMissCounts()

	space := filepath.Join(home, "spaces", "throwaway")
	kept := filepath.Join(home, "spaces", "keeper", "slapi")
	repos := []string{
		filepath.Join(space, "slapi"),
		filepath.Join(space, "console"),
	}
	for _, p := range append(append([]string{}, repos...), kept) {
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatal(err)
		}
		seedProject(t, p)
	}

	if err := os.RemoveAll(space); err != nil {
		t.Fatal(err)
	}

	// A single call purges everything missing right now — no debounce wait.
	purged := SweepDeletedProjectsNow()
	if len(purged) != len(repos) {
		t.Fatalf("purged=%v want all %d repos of the deleted space", purged, len(repos))
	}
	for _, p := range repos {
		assertPurged(t, p)
	}
	// The surviving space is untouched.
	if symbolCount(t, kept) == 0 {
		t.Fatalf("%s must not be purged", kept)
	}
	// No leftover debounce bookkeeping for paths that were just purged outright.
	if got := MissCount(repos[0]); got != 0 {
		t.Fatalf("MissCount after immediate purge=%d want 0", got)
	}
}

func TestProjectDataRequiresPath(t *testing.T) {
	if err := ProjectData("  "); err == nil {
		t.Fatal("expected error for empty project_path")
	}
}

func TestKnownProjectPathsIncludesIndexedButUnwatched(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(home, "git", "lonely")
	if err := os.MkdirAll(p, 0755); err != nil {
		t.Fatal(err)
	}
	seedProject(t, p)

	// Enumeration reads the index; a quiesced pool would legitimately return
	// nothing, so confirm the reader is available before asserting on it.
	if _, err := db.IndexReader(); err != nil {
		t.Skipf("index reader unavailable: %v", err)
	}
	found := false
	for _, got := range KnownProjectPaths() {
		if got == p {
			found = true
		}
	}
	if !found {
		t.Fatalf("KnownProjectPaths missing indexed-but-unwatched %s", p)
	}
}

// seedProject writes one row into every table a purge is expected to clear.
func seedProject(t *testing.T, projectPath string) {
	t.Helper()
	file := filepath.Join(projectPath, "a.go")
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('X','function',?,1,1,?)`, file, projectPath)
	db.IndexDB.Exec(`INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, 'fmt', 'import', ?)`, file, projectPath)
	db.IndexDB.Exec(`INSERT INTO indexed_files (file, project_path, indexed_at) VALUES (?, ?, datetime('now'))`, file, projectPath)
	db.IndexDB.Exec(`INSERT INTO summaries (file_path, symbol_name, summary_text, content_hash, project_path) VALUES (?, 'X', 'sum', 'h', ?)`, file, projectPath)
	db.IndexDB.Exec(`INSERT INTO vectors (content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (?, ?, 'code', ?, 'X', 'function', ?)`,
		"h-"+projectPath, []byte{1, 2, 3, 4}, file, projectPath)
	db.ContextDB.Exec(`INSERT INTO structured_memory (ref, kind, scope, project_path, subject, predicate, object) VALUES (?, 'fact', 'project', ?, 's', 'p', 'o')`,
		"mem-"+projectPath, projectPath)
	db.ContextDB.Exec(`INSERT INTO context_notes (ref, session_id, project_path, label, content, content_hash) VALUES (?, 'sess', ?, 'l', 'c', 'h')`,
		"note-"+projectPath, projectPath)
	db.DB.Exec(`INSERT INTO queries (tool_name, project_path) VALUES ('get_file_context', ?)`, projectPath)
	db.DB.Exec(`INSERT INTO sessions (session_id, file_path, symbol_name, start_line) VALUES ('sess', ?, 'X', 1)`, file)
}

func symbolCount(t *testing.T, projectPath string) int {
	t.Helper()
	var n int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM symbols WHERE project_path = ?`, projectPath).Scan(&n)
	return n
}

func assertPurged(t *testing.T, projectPath string) {
	t.Helper()
	index := map[string]string{
		"symbols":       `SELECT COUNT(*) FROM symbols WHERE project_path = ?`,
		"edges":         `SELECT COUNT(*) FROM edges WHERE project_path = ?`,
		"indexed_files": `SELECT COUNT(*) FROM indexed_files WHERE project_path = ?`,
		"summaries":     `SELECT COUNT(*) FROM summaries WHERE project_path = ?`,
		"vectors":       `SELECT COUNT(*) FROM vectors WHERE project_path = ?`,
	}
	for name, q := range index {
		var n int
		db.IndexDB.QueryRow(q, projectPath).Scan(&n)
		if n != 0 {
			t.Fatalf("%s: %s rows=%d want 0", projectPath, name, n)
		}
	}
	ctx := map[string]string{
		"structured_memory": `SELECT COUNT(*) FROM structured_memory WHERE project_path = ?`,
		"context_notes":     `SELECT COUNT(*) FROM context_notes WHERE project_path = ?`,
	}
	for name, q := range ctx {
		var n int
		db.ContextDB.QueryRow(q, projectPath).Scan(&n)
		if n != 0 {
			t.Fatalf("%s: %s rows=%d want 0", projectPath, name, n)
		}
	}
	var queries int
	db.DB.QueryRow(`SELECT COUNT(*) FROM queries WHERE project_path = ?`, projectPath).Scan(&queries)
	if queries != 0 {
		t.Fatalf("%s: queries rows=%d want 0", projectPath, queries)
	}
	var sessions int
	db.DB.QueryRow(`SELECT COUNT(*) FROM sessions WHERE file_path LIKE ?`, projectPath+"/%").Scan(&sessions)
	if sessions != 0 {
		t.Fatalf("%s: sessions rows=%d want 0", projectPath, sessions)
	}
}
