package projectmeta

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMarkClearWasDeleted(t *testing.T) {
	testExcludeDB(t)
	path := t.TempDir()

	if WasDeleted(path) {
		t.Fatal("fresh path should not be marked deleted")
	}

	MarkDeleted(path)
	if !WasDeleted(path) {
		t.Fatal("path should be marked deleted after MarkDeleted")
	}

	// Marking twice must not duplicate the entry or error.
	MarkDeleted(path)
	if !WasDeleted(path) {
		t.Fatal("path should still be marked deleted after a repeat MarkDeleted")
	}

	ClearDeleted(path)
	if WasDeleted(path) {
		t.Fatal("path should no longer be marked deleted after ClearDeleted")
	}

	// Clearing an already-clear path must not error or panic.
	ClearDeleted(path)
}

func TestDiscoverPathsSkipsDeleted(t *testing.T) {
	home := t.TempDir()
	gitRoot := filepath.Join(home, "git", "keep")
	deleted := filepath.Join(home, "git", "gone")
	if err := os.MkdirAll(filepath.Join(gitRoot, ".git"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(deleted, ".git"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	testExcludeDB(t)

	// The directory still exists on disk (only the index/data was purged), but it
	// was explicitly deleted from the dashboard, so passive discovery must skip it.
	MarkDeleted(deleted)

	paths := DiscoverPaths()
	for _, p := range paths {
		if p == filepath.Clean(deleted) {
			t.Fatalf("deleted path was re-discovered: %v", paths)
		}
	}
	found := false
	for _, p := range paths {
		if p == filepath.Clean(gitRoot) {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected keep repo in %v", paths)
	}

	// An explicit re-index (index_files, etc.) clears the tombstone, so discovery
	// picks the project back up.
	ClearDeleted(deleted)
	paths = DiscoverPaths()
	found = false
	for _, p := range paths {
		if p == filepath.Clean(deleted) {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected previously-deleted repo back in %v after ClearDeleted", paths)
	}
}

// Same as TestDiscoverPathsSkipsDeleted, but for the WTG-space layout specifically:
// deleting one repo checkout from the dashboard while its sibling repo (and the
// space directory itself) remain untouched on disk.
func TestDiscoverPathsSkipsDeletedSpaceRepoButKeepsSiblings(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	testExcludeDB(t)

	space := filepath.Join(home, "spaces", "echo")
	deletedRepo := filepath.Join(space, "slapi")
	keptRepo := filepath.Join(space, "console")
	mkWorktree(t, deletedRepo)
	mkWorktree(t, keptRepo)

	before := DiscoverPaths()
	foundDeleted, foundKept := false, false
	for _, p := range before {
		if p == filepath.Clean(deletedRepo) {
			foundDeleted = true
		}
		if p == filepath.Clean(keptRepo) {
			foundKept = true
		}
	}
	if !foundDeleted || !foundKept {
		t.Fatalf("both repos should be discovered before delete: %v", before)
	}

	// Simulate the dashboard's "delete project" action against just one repo in
	// the space — the directory itself (and its sibling) stay on disk.
	MarkDeleted(deletedRepo)
	if _, err := os.Stat(deletedRepo); err != nil {
		t.Fatalf("deleted repo's directory must still exist on disk: %v", err)
	}

	after := DiscoverPaths()
	for _, p := range after {
		if p == filepath.Clean(deletedRepo) {
			t.Fatalf("deleted space repo was re-discovered: %v", after)
		}
	}
	found := false
	for _, p := range after {
		if p == filepath.Clean(keptRepo) {
			found = true
		}
	}
	if !found {
		t.Fatalf("sibling repo in the same space must still be discovered: %v", after)
	}
}
