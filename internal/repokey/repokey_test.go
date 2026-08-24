package repokey

import (
	"os/exec"
	"path/filepath"
	"testing"
)

func gitInit(t *testing.T, dir string) {
	t.Helper()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Skipf("git unavailable: %v: %s", err, out)
		}
	}
	run("init", "-q", dir)
	run("-C", dir, "config", "user.email", "test@example.com")
	run("-C", dir, "config", "user.name", "test")
	run("-C", dir, "commit", "-q", "--allow-empty", "-m", "init")
}

func TestKeyGroupsWorktrees(t *testing.T) {
	Invalidate("")
	root := t.TempDir()
	main := filepath.Join(root, "main")
	linked := filepath.Join(root, "space", "repo")
	gitInit(t, main)

	cmd := exec.Command("git", "-C", main, "worktree", "add", "-q", "-b", "feature", linked)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("git worktree unavailable: %v: %s", err, out)
	}

	if !SameRepo(main, linked) {
		t.Fatalf("worktrees not grouped: %q vs %q", Key(main), Key(linked))
	}
}

func TestKeyDistinguishesUnrelatedRepos(t *testing.T) {
	Invalidate("")
	root := t.TempDir()
	a := filepath.Join(root, "a")
	b := filepath.Join(root, "b")
	gitInit(t, a)
	gitInit(t, b)
	if SameRepo(a, b) {
		t.Fatalf("unrelated repos share key %q", Key(a))
	}
}

func TestKeyFallsBackToPath(t *testing.T) {
	Invalidate("")
	dir := t.TempDir()
	if got := Key(dir); got == "" {
		t.Fatal("expected non-empty key for non-git dir")
	}
	if Key("") != "" {
		t.Fatal("empty path should yield empty key")
	}
}
