package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestHandleUpdateCheckRejectsPost(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/update/check", nil)
	w := httptest.NewRecorder()
	handleUpdateCheck(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleUpdateCheckReportsNotAGitCheckout(t *testing.T) {
	// os.Executable() under `go test` resolves to the compiled test binary's
	// own directory, not a git checkout — handleUpdateCheck should surface
	// that as a clean error in the JSON body rather than panic or hang.
	req := httptest.NewRequest(http.MethodGet, "/api/update/check", nil)
	w := httptest.NewRecorder()
	handleUpdateCheck(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var result map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if s, _ := result["error"].(string); s == "" {
		t.Fatalf("expected a non-empty 'error' field since the test binary's dir isn't a git checkout, got %+v", result)
	}
}

func TestHandleStartUpdateRejectsGet(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/update/start", nil)
	w := httptest.NewRecorder()
	handleStartUpdate(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleStartUpdateFailsCleanlyOutsideAGitCheckout(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/update/start", nil)
	w := httptest.NewRecorder()
	handleStartUpdate(w, req)
	if w.Code != http.StatusConflict {
		t.Fatalf("expected 409, got %d", w.Code)
	}
}

func TestHandleUpdateStatusRejectsPost(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/update/status", nil)
	w := httptest.NewRecorder()
	handleUpdateStatus(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleUpdateStatusReturnsSnapshot(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/update/status", nil)
	w := httptest.NewRecorder()
	handleUpdateStatus(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var body map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := body["active"]; !ok {
		t.Fatalf("expected an 'active' field, got %+v", body)
	}
}

func TestHandleRestartNowRejectsGet(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/restart-now", nil)
	w := httptest.NewRecorder()
	handleRestartNow(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

func TestHandleRestartNowFailsCleanlyWhenNotWiredUp(t *testing.T) {
	prev := db.RestartProcess
	db.RestartProcess = nil
	t.Cleanup(func() { db.RestartProcess = prev })

	req := httptest.NewRequest(http.MethodPost, "/api/restart-now", nil)
	w := httptest.NewRecorder()
	handleRestartNow(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestHandleRestartNowInvokesTheHook(t *testing.T) {
	prev := db.RestartProcess
	t.Cleanup(func() { db.RestartProcess = prev })
	called := make(chan struct{}, 1)
	db.RestartProcess = func() { called <- struct{}{} }

	req := httptest.NewRequest(http.MethodPost, "/api/restart-now", nil)
	w := httptest.NewRecorder()
	handleRestartNow(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	select {
	case <-called:
	case <-time.After(2 * time.Second):
		t.Fatal("expected RestartProcess to be invoked")
	}
}
