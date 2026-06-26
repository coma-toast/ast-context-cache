package indexer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func TestExpectedEmbedHashMatchesEmbedPath(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "a.go")
	if err := os.WriteFile(file, []byte("func f() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	got := ExpectedEmbedHash("function", "f", file, 1, 1)
	want := search.ContentHash("function f: func f() {}")
	if got != want {
		t.Fatalf("hash=%q want %q", got, want)
	}
}

func TestReindexDeletesCodeVectors(t *testing.T) {
	project := "reindex-vec-test"
	file := filepath.Join(t.TempDir(), "x.go")
	if err := os.WriteFile(file, []byte("func g(){}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	db.IndexDB.Exec(`DELETE FROM vectors WHERE project_path = ?`, project)
	db.IndexDB.Exec(`DELETE FROM symbols WHERE project_path = ?`, project)
	tx, err := db.IndexDB.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err := deleteCodeVectorsTx(tx, file, project); err != nil {
		t.Fatal(err)
	}
	tx.Exec(`DELETE FROM symbols WHERE file = ? AND project_path = ?`, file, project)
	res, err := tx.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, embed_hash) VALUES ('g', 'function', ?, 1, 1, '', 'x.g', ?, 'h1')`, file, project)
	if err != nil {
		t.Fatal(err)
	}
	symID, _ := res.LastInsertId()
	tx.Exec(`INSERT INTO vectors (symbol_id, content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (?, 'h0', ?, 'code', ?, 'g', 'function', ?)`, symID, []byte{9}, file, project)
	tx.Commit()
	tx2, _ := db.IndexDB.Begin()
	if err := deleteCodeVectorsTx(tx2, file, project); err != nil {
		t.Fatal(err)
	}
	tx2.Exec(`DELETE FROM symbols WHERE file = ? AND project_path = ?`, file, project)
	tx2.Commit()
	var count int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM vectors WHERE source_file = ? AND project_path = ?`, file, project).Scan(&count)
	if count != 0 {
		t.Fatalf("vectors=%d want 0 after reindex delete", count)
	}
}
