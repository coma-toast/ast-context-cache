package dashboard

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMetricsEndpoint(t *testing.T) {
	h := NewHandler("")
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("GET /metrics status=%d body=%s", rr.Code, rr.Body.String())
	}
	body := rr.Body.String()
	if !strings.Contains(body, "astcache_") {
		t.Fatalf("expected astcache_ metrics in body, got %q", body[:min(200, len(body))])
	}
	for _, name := range []string{
		"astcache_up",
		"astcache_embed_pending",
		"astcache_embed_queued",
		"astcache_embed_in_flight",
		"astcache_embed_workers_target",
		"astcache_index_wal_bytes",
		"astcache_tokens_saved_today",
		"astcache_embedder_state",
	} {
		if !strings.Contains(body, name) {
			t.Errorf("missing metric %s", name)
		}
	}
}
