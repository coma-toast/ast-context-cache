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
		`astcache_ledger_tokens_saved_today{ledger="compression"}`,
		`astcache_ledger_tokens_saved_today{ledger="dedup"}`,
		"astcache_embedder_state",
		"astcache_query_cache_hit_ratio",
		"astcache_handoffs_created_total",
		"astcache_handoff_children_opened_total",
		"astcache_handoff_children_resumed_total",
		`astcache_handoff_children_completed_total{status="partial"}`,
		"astcache_handoff_children_abandoned_total",
		`astcache_handoff_child_searches_total{repeat="true"}`,
		"astcache_handoff_trees_expired_total",
		"astcache_handoff_open_trees",
		"astcache_handoff_open_children",
		"astcache_handoff_repeat_search_ratio",
		"astcache_handoff_tree_tokens",
		"astcache_handoff_claim_wait_seconds",
	} {
		if !strings.Contains(body, name) {
			t.Errorf("missing metric %s", name)
		}
	}
}
