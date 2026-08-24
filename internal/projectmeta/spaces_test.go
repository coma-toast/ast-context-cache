package projectmeta

import (
	"os"
	"path/filepath"
	"testing"
)

// mkRepo creates a checkout whose .git is a directory (a plain clone).
func mkRepo(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, ".git"), 0755); err != nil {
		t.Fatal(err)
	}
}

// mkWorktree creates a checkout whose .git is a FILE, as WTG worktrees are.
func mkWorktree(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".git"), []byte("gitdir: /elsewhere/.git/worktrees/x\n"), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestListSpaceRepoPaths(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	space := filepath.Join(home, "spaces", "echo")

	mkWorktree(t, filepath.Join(space, "slapi"))
	mkWorktree(t, filepath.Join(space, "console"))
	mkRepo(t, filepath.Join(space, "sandbox"))
	// Not repos: a plain directory, a hidden directory, and a loose file.
	if err := os.MkdirAll(filepath.Join(space, "notes"), 0755); err != nil {
		t.Fatal(err)
	}
	mkWorktree(t, filepath.Join(space, ".hidden"))
	if err := os.WriteFile(filepath.Join(space, "README"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	got, err := ListSpaceRepoPaths("echo")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		filepath.Join(space, "console"),
		filepath.Join(space, "sandbox"),
		filepath.Join(space, "slapi"),
	}
	if len(got) != len(want) {
		t.Fatalf("got=%v want=%v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%q want %q (all: %v)", i, got[i], want[i], got)
		}
	}
}

// A space listing must not depend on the repo having ever been indexed or watched.
func TestListSpaceRepoPathsIncludesNeverIndexed(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	fresh := filepath.Join(home, "spaces", "foxtrot", "brand-new")
	mkWorktree(t, fresh)

	got, err := ListSpaceRepoPaths("foxtrot")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != fresh {
		t.Fatalf("got=%v want [%s]", got, fresh)
	}
}

func TestListSpaceRepoPathsRejectsBadNames(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	mkWorktree(t, filepath.Join(home, "spaces", "echo", "slapi"))

	for _, name := range []string{"", "  ", ".", "..", "../echo", "echo/slapi", "/etc"} {
		if _, err := ListSpaceRepoPaths(name); err == nil {
			t.Fatalf("expected error for space name %q", name)
		}
	}
	if _, err := ListSpaceRepoPaths("nope"); err == nil {
		t.Fatal("expected error for missing space")
	}
}

func TestSpacesRootDefaultsToHomeSpaces(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if got, want := SpacesRoot(), filepath.Join(home, "spaces"); got != want {
		t.Fatalf("SpacesRoot=%q want %q", got, want)
	}
}
