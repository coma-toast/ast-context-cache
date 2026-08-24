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

type startSpaceResponse struct {
	Status         string   `json:"status"`
	Space          string   `json:"space"`
	Started        []string `json:"started"`
	AlreadyRunning []string `json:"already_running"`
	Skipped        []string `json:"skipped"`
	Errors         []string `json:"errors"`
	Error          string   `json:"error"`
}

func postSpace(t *testing.T, body string) (*httptest.ResponseRecorder, startSpaceResponse) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/start-watcher-space", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handleStartWatcherSpace(rec, req)
	var out startSpaceResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal %s: %v", rec.Body.String(), err)
	}
	return rec, out
}

func mkSpaceWorktree(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".git"), []byte("gitdir: /elsewhere\n"), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestHandleStartWatcherSpaceStartsEveryRepo(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	space := filepath.Join(home, "spaces", "echo")
	repoA := filepath.Join(space, "slapi")
	repoB := filepath.Join(space, "console")
	mkSpaceWorktree(t, repoA)
	mkSpaceWorktree(t, repoB)
	t.Cleanup(func() {
		watcher.DeleteWatcher(repoA)
		watcher.DeleteWatcher(repoB)
	})

	rec, out := postSpace(t, `{"space":"echo"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body.String())
	}
	if out.Space != "echo" || out.Status != "started" {
		t.Fatalf("out=%+v", out)
	}
	if len(out.Started) != 2 {
		t.Fatalf("started=%v want both repos", out.Started)
	}
	for _, p := range []string{repoA, repoB} {
		if !watcher.IsActive(p) {
			t.Fatalf("watcher not active for %s", p)
		}
	}

	// A second call is idempotent: nothing new started, both already running.
	_, again := postSpace(t, `{"space":"echo"}`)
	if len(again.Started) != 0 || len(again.AlreadyRunning) != 2 {
		t.Fatalf("second call: started=%v already=%v", again.Started, again.AlreadyRunning)
	}
}

func TestHandleStartWatcherSpaceValidates(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	rec, out := postSpace(t, `{"space":""}`)
	if rec.Code != http.StatusBadRequest || out.Error == "" {
		t.Fatalf("empty space: code=%d out=%+v", rec.Code, out)
	}
	rec, out = postSpace(t, `{"space":"missing"}`)
	if rec.Code != http.StatusBadRequest || out.Error == "" {
		t.Fatalf("missing space: code=%d out=%+v", rec.Code, out)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/start-watcher-space", nil)
	rec = httptest.NewRecorder()
	handleStartWatcherSpace(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET code=%d want 405", rec.Code)
	}
}

func TestParseSpaceFromRequest(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(`{"space":" echo "}`))
	req.Header.Set("Content-Type", "application/json")
	if got := parseSpaceFromRequest(req); got != "echo" {
		t.Fatalf("json space=%q", got)
	}
	req = httptest.NewRequest(http.MethodPost, "/x", strings.NewReader("space=bravo"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if got := parseSpaceFromRequest(req); got != "bravo" {
		t.Fatalf("form space=%q", got)
	}
}
