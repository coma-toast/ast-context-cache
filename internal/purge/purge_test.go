package purge

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// Deleting a project must not let it silently reappear: any active watcher has to
// stop immediately (otherwise the next file-save event under the directory
// re-indexes it with zero explicit tool calls), a pinned project must be unpinned
// (otherwise the next ast-mcp restart auto-watches and re-indexes it), and the path
// must be tombstoned so passive filesystem discovery doesn't re-list it either.
func TestProjectDataStopsReappearing(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	p := filepath.Join(home, "git", "deleteme")
	if err := os.MkdirAll(p, 0755); err != nil {
		t.Fatal(err)
	}
	seedProject(t, p)

	if err := db.TogglePinnedProject(p, true); err != nil {
		t.Fatal(err)
	}
	watcher.StartWatcher(p)
	t.Cleanup(func() { watcher.DeleteWatcher(p) })
	if !watcher.IsActive(p) {
		t.Fatal("watcher should be active before delete")
	}

	if err := ProjectData(p); err != nil {
		t.Fatal(err)
	}

	if watcher.IsActive(p) {
		t.Fatal("watcher must be stopped once the project is deleted")
	}
	if db.IsPinnedProject(p) {
		t.Fatal("project must be unpinned once deleted, or it auto-reindexes on next restart")
	}
	if !projectmeta.WasDeleted(p) {
		t.Fatal("project must be tombstoned so passive discovery doesn't re-list it")
	}
}

// A deleted project's parent/child monorepo-container link must not survive the
// delete. If it did, using the deleted child's path as a *parent* for some new,
// unrelated link later would be wrongly refused: validateLink checks Parents(newParent)
// and blocks with "parent is already linked under another container" the moment any
// row — stale or not — still names that path as a child.
func TestProjectDataRemovesLinks(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	parent := filepath.Join(home, "git", "monorepo")
	child := filepath.Join(parent, "service")
	if err := os.MkdirAll(child, 0755); err != nil {
		t.Fatal(err)
	}
	seedProject(t, parent)
	seedProject(t, child)
	if err := projectlinks.CreateLink(parent, child, false); err != nil {
		t.Fatal(err)
	}

	if err := ProjectData(child); err != nil {
		t.Fatal(err)
	}

	linked, err := projectlinks.Links(parent)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range linked {
		if c == projectlinks.NormalizePath(child) {
			t.Fatalf("deleted child still linked under parent: %v", linked)
		}
	}
	if parents, err := projectlinks.Parents(child); err != nil || len(parents) != 0 {
		t.Fatalf("Parents(child)=%v err=%v want empty", parents, err)
	}

	// The deleted child's path must be free to become a parent of its own new,
	// unrelated sub-project — a leftover link row would incorrectly refuse this.
	grandchild := filepath.Join(child, "sub")
	if err := os.MkdirAll(grandchild, 0755); err != nil {
		t.Fatal(err)
	}
	if err := projectlinks.CreateLink(child, grandchild, false); err != nil {
		t.Fatalf("using deleted child's path as a new parent should succeed: %v", err)
	}
}
