package purge

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
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
