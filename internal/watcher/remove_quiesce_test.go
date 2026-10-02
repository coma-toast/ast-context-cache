package watcher

import (
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Removing a deleted file from the index takes its code vectors in the same
// transaction as its symbols. While a WAL quiesce gates index writes, nothing may
// be deleted, so the file stays recorded as indexed for the next catch-up scan.
func TestRemoveFileFromIndexIsAllOrNothing(t *testing.T) {
	t.Cleanup(func() { db.SetIndexReadGateForTest(false) })
	project := filepath.Join(t.TempDir(), "proj")
	file := filepath.Join(project, "gone.go")
	other := filepath.Join(project, "kept.go")
	for _, f := range []string{file, other} {
		mustExec(t, `INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('X', 'function', ?, 1, 1, ?)`, f, project)
		mustExec(t, `INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, 'fmt', 'import', ?)`, f, project)
		mustExec(t, `INSERT INTO indexed_files (file, project_path, indexed_at) VALUES (?, ?, datetime('now'))`, f, project)
		mustExec(t, `INSERT INTO vectors (content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (?, ?, 'code', ?, 'X', 'function', ?)`,
			"h-"+f, []byte{1, 2, 3, 4}, f, project)
	}

	db.SetIndexReadGateForTest(true)
	if err := removeFileFromIndex(file, project); err == nil {
		t.Fatal("removal succeeded while the index was quiesced")
	}
	db.SetIndexReadGateForTest(false)
	if got := fileRows(t, file, project); got != [4]int{1, 1, 1, 1} {
		t.Fatalf("after failed removal: symbols/edges/indexed_files/vectors=%v, want all kept", got)
	}

	if err := removeFileFromIndex(file, project); err != nil {
		t.Fatal(err)
	}
	if got := fileRows(t, file, project); got != [4]int{} {
		t.Fatalf("after removal: symbols/edges/indexed_files/vectors=%v, want none", got)
	}
	if got := fileRows(t, other, project); got != [4]int{1, 1, 1, 1} {
		t.Fatalf("other file: symbols/edges/indexed_files/vectors=%v, want all kept", got)
	}
}

func mustExec(t *testing.T, q string, args ...any) {
	t.Helper()
	if _, err := db.IndexDB.Exec(q, args...); err != nil {
		t.Fatal(err)
	}
}

func fileRows(t *testing.T, file, project string) [4]int {
	t.Helper()
	var out [4]int
	for i, q := range []string{
		`SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`,
		`SELECT COUNT(*) FROM edges WHERE source_file = ? AND project_path = ?`,
		`SELECT COUNT(*) FROM indexed_files WHERE file = ? AND project_path = ?`,
		`SELECT COUNT(*) FROM vectors WHERE source_file = ? AND project_path = ?`,
	} {
		if err := db.IndexDB.QueryRow(q, file, project).Scan(&out[i]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}
