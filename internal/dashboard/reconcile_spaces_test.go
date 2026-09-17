package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// The Spaces card's "Refresh" button hits this endpoint to reconcile
// ast-context-cache's indexed projects against what's actually still on
// disk under a WTG space — with no confirmation, since spaces are ephemeral
// by design.
func TestHandleReconcileSpacesPurgesMissingSpaceRepoOnly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	spaceRepo := filepath.Join(home, "spaces", "throwaway", "slapi")
	if err := os.MkdirAll(spaceRepo, 0755); err != nil {
		t.Fatal(err)
	}
	watcher.StartWatcher(spaceRepo)
	t.Cleanup(func() { watcher.DeleteWatcher(spaceRepo) })
	if !watcher.IsActive(spaceRepo) {
		t.Fatal("watcher should be active before the space is removed")
	}

	if err := os.RemoveAll(filepath.Join(home, "spaces")); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/reconcile-spaces", nil)
	rec := httptest.NewRecorder()
	handleReconcileSpaces(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}

	var out struct {
		Status string   `json:"status"`
		Purged []string `json:"purged"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v (body=%s)", err, rec.Body.String())
	}
	if out.Status != "reconciled" {
		t.Fatalf("status=%q want reconciled", out.Status)
	}
	found := false
	for _, p := range out.Purged {
		if strings.HasSuffix(p, "slapi") {
			found = true
		}
	}
	if !found {
		t.Fatalf("purged=%v want it to include the removed space repo", out.Purged)
	}
	if watcher.IsActive(spaceRepo) {
		t.Fatal("watcher for the removed space repo should have been stopped")
	}
}

func TestHandleReconcileSpacesRejectsGet(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/reconcile-spaces", nil)
	rec := httptest.NewRecorder()
	handleReconcileSpaces(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("code=%d want 405", rec.Code)
	}
}
