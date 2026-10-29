package watcher

import (
	"os"
	"path/filepath"
	"testing"
)

// JD-254: a watcher started through a symlinked project root is keyed under the
// link's target, so its catch-up walk enters the directory instead of visiting
// the link as one non-directory entry, and either spelling finds the watcher.
func TestEnsureWatcherThroughSymlinkedRoot(t *testing.T) {
	prevHook := PostIndexHook
	PostIndexHook = nil
	t.Cleanup(func() { PostIndexHook = prevHook })
	base := NormalizeProjectPath(t.TempDir())
	real, link := filepath.Join(base, "real"), filepath.Join(base, "link")
	writeGoFile(t, filepath.Join(real, "a.go"), "A")
	writeGoFile(t, filepath.Join(real, "sub", "b.go"), "B")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	if got := NormalizeProjectPath(link); got != real {
		t.Fatalf("NormalizeProjectPath(link) = %q, want %q", got, real)
	}
	t.Cleanup(func() { DeleteWatcher(real) })
	cleanupWatchers(t)
	EnsureWatcher(link)
	for _, p := range []string{link, real} {
		if !IsActive(p) {
			t.Fatalf("IsActive(%s) = false", p)
		}
	}
	waitFiles(t, real, 2)
	if n := rowCount(t, `SELECT COUNT(*) FROM symbols WHERE project_path = ?`, link); n != 0 {
		t.Fatalf("%d rows keyed under the link spelling, want 0", n)
	}
}
