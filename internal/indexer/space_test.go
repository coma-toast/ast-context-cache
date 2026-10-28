package indexer

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// gitSpace builds a main clone with the given files committed, and a wtg-style
// space holding one worktree per name (each on its own branch) plus go.work.
func gitSpace(t *testing.T, files map[string]string, worktrees ...string) (mainRepo, space string) {
	t.Helper()
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	root := t.TempDir()
	mainRepo = filepath.Join(root, "git", "repo")
	space = filepath.Join(root, "spaces", "feature")
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Skipf("git unavailable: %v: %s", err, out)
		}
	}
	if err := os.MkdirAll(mainRepo, 0o755); err != nil {
		t.Fatal(err)
	}
	git("init", "-q", mainRepo)
	git("-C", mainRepo, "config", "user.email", "test@example.com")
	git("-C", mainRepo, "config", "user.name", "test")
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(mainRepo, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("-C", mainRepo, "add", "-A")
	git("-C", mainRepo, "commit", "-q", "-m", "init")
	if err := os.MkdirAll(space, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(space, "go.work"), []byte("go 1.23\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, wt := range worktrees {
		git("-C", mainRepo, "worktree", "add", "-q", "-b", "space-"+wt, filepath.Join(space, wt))
	}
	return projectlinks.NormalizePath(mainRepo), projectlinks.NormalizePath(space)
}

func projectRows(t *testing.T, projectPath string) (symbols, files int) {
	t.Helper()
	if err := db.IndexDB.QueryRow(`SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?`, projectPath).Scan(&symbols, &files); err != nil {
		t.Fatal(err)
	}
	return symbols, files
}

// A space root is indexed as its checkouts: each is its own project linked under
// the root, unchanged files are copied from the indexed main clone, and the root
// itself gets no rows.
func TestIndexDirectorySpaceRootIndexesCheckouts(t *testing.T) {
	files := map[string]string{
		"a.go": "package p\n\nfunc Alpha() {}\n",
		"b.go": "package p\n\nfunc Bravo() {}\n",
	}
	mainRepo, space := gitSpace(t, files, "repo", "other")
	if _, err := IndexDirectory(mainRepo, mainRepo); err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(space, "repo")
	other := filepath.Join(space, "other")
	// One file differs from the main clone, so it must be parsed, not copied.
	if err := os.WriteFile(filepath.Join(repo, "b.go"), []byte("package p\n\nfunc Changed() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	reused, parsed := 0, 0
	src := FindReuseSource(repo)
	if src == nil || src.ProjectPath != mainRepo {
		t.Fatalf("reuse source=%v want %s", src, mainRepo)
	}
	for _, f := range []string{"a.go", "b.go"} {
		out, err := IndexOrReuse(filepath.Join(repo, f), repo, src)
		if err != nil {
			t.Fatal(err)
		}
		if out.Reused {
			reused++
		} else {
			parsed++
		}
	}
	t.Logf("checkout %s: reused=%d parsed=%d", repo, reused, parsed)
	if reused != 1 || parsed != 1 {
		t.Fatalf("reused=%d parsed=%d want 1/1", reused, parsed)
	}

	if _, err := IndexDirectory(space, space); err != nil {
		t.Fatal(err)
	}
	if n, _ := projectRows(t, space); n != 0 {
		t.Fatalf("space root has %d symbols, want none", n)
	}
	for _, p := range []string{repo, other} {
		if _, f := projectRows(t, p); f != 2 {
			t.Fatalf("%s indexed %d files, want 2", p, f)
		}
	}
	var name string
	db.IndexDB.QueryRow(`SELECT name FROM symbols WHERE project_path = ? AND file = ?`, repo, filepath.Join(repo, "b.go")).Scan(&name)
	if name != "Changed" {
		t.Fatalf("changed file symbol=%q want Changed", name)
	}
	links, _ := projectlinks.Links(space)
	if len(links) != 2 {
		t.Fatalf("links=%v want both checkouts", links)
	}
	if scope := projectlinks.ResolveScope(space); len(scope) != 3 {
		t.Fatalf("scope=%v want root plus two checkouts", scope)
	}

	if _, err := IndexDirectory(filepath.Join(space, "repo"), space); err != nil {
		t.Fatalf("subdirectory of a checkout: %v", err)
	}
	loose := filepath.Join(space, "notes")
	if err := os.MkdirAll(loose, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := IndexDirectory(loose, space); err == nil {
		t.Fatal("expected an error for a space directory outside every checkout")
	}
}
