package indexer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Both IndexFishFile and indexPlaintextFile were rewritten to route their
// writes through db.IndexWrite instead of a raw conn.Begin() on the reader
// pool. Neither had any prior test coverage, so these confirm the rewrite
// still indexes correctly end-to-end.
func TestIndexFishFile(t *testing.T) {
	dir := t.TempDir()
	projectPath := dir
	file := filepath.Join(dir, "helper.fish")
	script := "function greet\n    echo hello\nend\n"
	if err := os.WriteFile(file, []byte(script), 0644); err != nil {
		t.Fatal(err)
	}

	count, _, _, err := IndexFishFile(file, projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("count=%d want 1", count)
	}

	var name, kind string
	if err := db.IndexDB.QueryRow(`SELECT name, kind FROM symbols WHERE file = ? AND project_path = ?`, file, projectPath).Scan(&name, &kind); err != nil {
		t.Fatal(err)
	}
	if name != "greet" || kind != "function" {
		t.Fatalf("name=%q kind=%q want greet/function", name, kind)
	}

	var indexedAt string
	if err := db.IndexDB.QueryRow(`SELECT indexed_at FROM indexed_files WHERE file = ? AND project_path = ?`, file, projectPath).Scan(&indexedAt); err != nil {
		t.Fatalf("indexed_files row missing: %v", err)
	}
}

func TestIndexPlaintextFile(t *testing.T) {
	dir := t.TempDir()
	projectPath := dir
	file := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(file, []byte("first line\nsecond line\n"), 0644); err != nil {
		t.Fatal(err)
	}

	count, fullTokens, _, err := indexPlaintextFile(file, projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("count=%d want 1", count)
	}
	if fullTokens <= 0 {
		t.Fatalf("fullTokens=%d want > 0", fullTokens)
	}

	var kind, skeleton string
	if err := db.IndexDB.QueryRow(`SELECT kind, skeleton FROM symbols WHERE file = ? AND project_path = ?`, file, projectPath).Scan(&kind, &skeleton); err != nil {
		t.Fatal(err)
	}
	if kind != "plaintext" || skeleton != "first line" {
		t.Fatalf("kind=%q skeleton=%q want plaintext/%q", kind, skeleton, "first line")
	}
}
