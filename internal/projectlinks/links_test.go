package projectlinks

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestIsStrictSubpath(t *testing.T) {
	parent := "/Users/j/git"
	if !IsStrictSubpath("/Users/j/git/foo", parent) {
		t.Fatal("expected strict subpath")
	}
	if IsStrictSubpath("/Users/j/git", parent) {
		t.Fatal("equal path is not strict subpath")
	}
	if IsStrictSubpath("/Users/j/other", parent) {
		t.Fatal("sibling is not subpath")
	}
}

func TestCreateLinkAndScope(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, "git")
	child := filepath.Join(parent, "foo")
	if err := os.MkdirAll(child, 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", root)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if err := CreateLink(parent, child, false); err != nil {
		t.Fatal(err)
	}
	scope := ResolveScope(parent)
	if len(scope) != 2 {
		t.Fatalf("scope=%v want 2 entries", scope)
	}
	if !IsUnderLinkedChild(filepath.Join(child, "main.go"), parent) {
		t.Fatal("file under linked child should skip")
	}
	if err := Unlink(parent, child); err != nil {
		t.Fatal(err)
	}
	if IsUnderLinkedChild(filepath.Join(child, "main.go"), parent) {
		t.Fatal("after unlink should not skip")
	}
}

func TestScopeSQL(t *testing.T) {
	frag, args := ScopeSQL("s", "/tmp/parent")
	if frag != "s.project_path = ?" || len(args) != 1 {
		t.Fatalf("single scope: frag=%q args=%v", frag, args)
	}
}

func TestOwningProject(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, "git")
	child := filepath.Join(parent, "foo")
	if err := os.MkdirAll(child, 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", root)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if err := CreateLink(parent, child, false); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(child, "a.go")
	if got := OwningProject(file, parent); got != child {
		t.Fatalf("OwningProject=%q want %q", got, child)
	}
}
