package dashboard

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// The Watchers panel's delete button (handleDeleteWatcher -> deleteProjectData)
// used to duplicate purge.ProjectData's cleanup by hand and had drifted from it:
// it never un-pinned or tombstoned the project, so a pinned project deleted from
// here (unlike the Settings tab's delete button) stayed pinned and would reappear
// on the next ast-mcp restart. deleteProjectData now delegates to purge.ProjectData
// so both delete entry points behave identically.
func TestHandleDeleteWatcherStopsReappearing(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	p := filepath.Join(home, "git", "deleteme")
	if err := os.MkdirAll(p, 0755); err != nil {
		t.Fatal(err)
	}
	if err := db.TogglePinnedProject(p, true); err != nil {
		t.Fatal(err)
	}
	watcher.StartWatcher(p)
	t.Cleanup(func() { watcher.DeleteWatcher(p) })
	if !watcher.IsActive(p) {
		t.Fatal("watcher should be active before delete")
	}

	req := httptest.NewRequest(http.MethodPost, "/api/delete-watcher", strings.NewReader(`{"project_path":"`+p+`"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handleDeleteWatcher(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}

	if watcher.IsActive(p) {
		t.Fatal("watcher must be stopped once deleted via the Watchers panel")
	}
	if db.IsPinnedProject(p) {
		t.Fatal("project must be unpinned once deleted via the Watchers panel")
	}
	if !projectmeta.WasDeleted(p) {
		t.Fatal("project must be tombstoned once deleted via the Watchers panel")
	}
}
