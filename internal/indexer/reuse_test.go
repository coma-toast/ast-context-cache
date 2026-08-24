package indexer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestReuseFileCopiesSiblingRows(t *testing.T) {
	root := t.TempDir()
	sibling := filepath.Join(root, "alpha", "repo")
	fresh := filepath.Join(root, "bravo", "repo")
	if err := os.MkdirAll(sibling, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(fresh, 0755); err != nil {
		t.Fatal(err)
	}
	src := "package p\n\nimport \"fmt\"\n\nfunc Hello() { fmt.Println(\"hi\") }\n"
	sibFile := filepath.Join(sibling, "a.go")
	newFile := filepath.Join(fresh, "a.go")
	if err := os.WriteFile(sibFile, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newFile, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := IndexFile(sibFile, sibling); err != nil {
		t.Fatal(err)
	}

	n, ok := ReuseFile(newFile, fresh, &ReuseSource{ProjectPath: sibling})
	if !ok {
		t.Fatal("expected reuse for byte-identical file")
	}
	if n == 0 {
		t.Fatal("expected copied symbols")
	}

	var name, kind, embedHash string
	err := db.IndexDB.QueryRow(`SELECT name, kind, COALESCE(embed_hash,'') FROM symbols WHERE file = ? AND project_path = ?`,
		newFile, fresh).Scan(&name, &kind, &embedHash)
	if err != nil {
		t.Fatal(err)
	}
	if name != "Hello" || kind != "function" {
		t.Fatalf("symbol=%s/%s want Hello/function", name, kind)
	}
	// Content-derived, so it must match what a real parse of the new file yields.
	if want := ExpectedEmbedHash(kind, name, newFile, 5, 5); embedHash != want {
		t.Fatalf("embed_hash=%q want %q", embedHash, want)
	}

	var edges int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM edges WHERE source_file = ? AND project_path = ?`, newFile, fresh).Scan(&edges)
	if edges == 0 {
		t.Fatal("expected import edges copied")
	}
	var indexed int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM indexed_files WHERE file = ? AND project_path = ?`, newFile, fresh).Scan(&indexed)
	if indexed != 1 {
		t.Fatalf("indexed_files=%d want 1", indexed)
	}
}

func TestReuseFileSkipsChangedContent(t *testing.T) {
	root := t.TempDir()
	sibling := filepath.Join(root, "alpha", "repo")
	fresh := filepath.Join(root, "bravo", "repo")
	os.MkdirAll(sibling, 0755)
	os.MkdirAll(fresh, 0755)
	sibFile := filepath.Join(sibling, "b.go")
	newFile := filepath.Join(fresh, "b.go")
	if err := os.WriteFile(sibFile, []byte("func A() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newFile, []byte("func B() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := IndexFile(sibFile, sibling); err != nil {
		t.Fatal(err)
	}
	if _, ok := ReuseFile(newFile, fresh, &ReuseSource{ProjectPath: sibling}); ok {
		t.Fatal("expected no reuse when content differs")
	}
	if _, ok := ReuseFile(newFile, fresh, nil); ok {
		t.Fatal("expected no reuse without a source")
	}
}

func TestReuseFileSkipsMissingSibling(t *testing.T) {
	root := t.TempDir()
	sibling := filepath.Join(root, "alpha", "repo")
	fresh := filepath.Join(root, "bravo", "repo")
	os.MkdirAll(sibling, 0755)
	os.MkdirAll(fresh, 0755)
	newFile := filepath.Join(fresh, "c.go")
	if err := os.WriteFile(newFile, []byte("func C() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, ok := ReuseFile(newFile, fresh, &ReuseSource{ProjectPath: sibling}); ok {
		t.Fatal("expected no reuse when sibling has no such file")
	}
}
