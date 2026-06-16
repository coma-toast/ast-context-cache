package embedder

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestTestSettings_remoteProbeTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(20 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	res := TestSettings(Settings{Backend: "http", HTTPURL: srv.URL}, "")
	if res.OK {
		t.Fatal("expected probe failure")
	}
	if !strings.Contains(strings.ToLower(res.Error), "timeout") &&
		!strings.Contains(strings.ToLower(res.Error), "deadline") {
		t.Fatalf("expected timeout error, got %q", res.Error)
	}
	if res.LatencyMs > int64(ProbeTimeout/time.Millisecond)+2000 {
		t.Fatalf("probe took too long: %dms", res.LatencyMs)
	}
}
