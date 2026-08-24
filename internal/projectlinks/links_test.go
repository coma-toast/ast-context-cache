package projectlinks

import (
	"os"
	"os/exec"
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

func TestResolveScopeWithRepoSiblings(t *testing.T) {
	root := t.TempDir()
	main := filepath.Join(root, "git", "repo")
	linked := filepath.Join(root, "space", "repo")
	if err := os.MkdirAll(main, 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", root)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	gitRun := func(args ...string) {
		if out, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Skipf("git unavailable: %v: %s", err, out)
		}
	}
	gitRun("init", "-q", main)
	gitRun("-C", main, "config", "user.email", "test@example.com")
	gitRun("-C", main, "config", "user.name", "test")
	gitRun("-C", main, "commit", "-q", "--allow-empty", "-m", "init")
	gitRun("-C", main, "worktree", "add", "-q", "-b", "feature", linked)

	// Only indexed projects count as siblings.
	if scope := ResolveScopeWithRepoSiblings(main, true); len(scope) != 1 {
		t.Fatalf("scope=%v want only self before sibling is indexed", scope)
	}
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('X','function',?,1,1,?)`,
		filepath.Join(linked, "a.go"), linked)

	scope := ResolveScopeWithRepoSiblings(main, true)
	if len(scope) != 2 || scope[0] != NormalizePath(main) {
		t.Fatalf("scope=%v want self plus sibling", scope)
	}
	if scope[1] != NormalizePath(linked) {
		t.Fatalf("sibling=%q want %q", scope[1], NormalizePath(linked))
	}
	if base := ResolveScopeWithRepoSiblings(main, false); len(base) != 1 {
		t.Fatalf("opt-out scope=%v want self only", base)
	}
	if base := ResolveScope(main); len(base) != 1 {
		t.Fatalf("ResolveScope must stay sibling-free, got %v", base)
	}

	frag, args, used := ScopeSQLWithRepoSiblings("s", main, true)
	if frag != "s.project_path IN (?,?)" || len(args) != 2 || len(used) != 2 {
		t.Fatalf("frag=%q args=%v used=%v", frag, args, used)
	}
	frag, args, used = ScopeSQLWithRepoSiblings("", main, false)
	if frag != "project_path = ?" || len(args) != 1 || len(used) != 1 {
		t.Fatalf("opt-out frag=%q args=%v used=%v", frag, args, used)
	}
}
