package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestComputeValueHeuristic(t *testing.T) {
	h := computeValueHeuristic(8000, 2000, 10000, 30)
	if !h.HeuristicApproximate || h.HeuristicLabel != "approximate" {
		t.Fatalf("expected approximate label, got %+v", h)
	}
	if h.ApproxBaselineTokens != 10000 {
		t.Fatalf("baseline=%d", h.ApproxBaselineTokens)
	}
	if h.ApproxTokensReturned != 2000 {
		t.Fatalf("returned=%d", h.ApproxTokensReturned)
	}
	if h.ApproxRoundsAvoided != 2.0 {
		t.Fatalf("rounds=%v want 2", h.ApproxRoundsAvoided)
	}
	h2 := computeValueHeuristic(4000, 1000, 0, 7)
	if h2.ApproxBaselineTokens != 5000 {
		t.Fatalf("fallback baseline=%d", h2.ApproxBaselineTokens)
	}
	if h2.ApproxRoundsAvoided != 1.0 {
		t.Fatalf("rounds=%v want 1", h2.ApproxRoundsAvoided)
	}
}

func TestWeeklyDigestAndContextSessionsAPI(t *testing.T) {
	testEmbedDB(t)
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := db.DB.Exec(`INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, tokens_used, symbol_baseline_tokens, duration_ms)
		VALUES (?, 'get_context_capsule', 'sess-a', '/proj', 4000, 1000, 5000, 12.5)`, now)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.DB.Exec(`INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, tokens_used, duration_ms)
		VALUES (?, 'store_context', 'sess-a', '/proj', 800, 0, 5)`, now)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.DB.Exec(`INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, tokens_used, duration_ms)
		VALUES (?, 'fetch_context', 'sess-a', '/proj', 0, 750, 4)`, now)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.DB.Exec(`INSERT INTO context_session_stats (session_id, project_path, notes_count, virtual_tokens_stored, virtual_tokens_accessed, last_store_at, last_access_at)
		VALUES ('sess-a', '/proj', 2, 800, 750, ?, ?)`, now, now)
	if err != nil {
		t.Fatal(err)
	}
	if db.ContextDB != nil {
		_, _ = db.ContextDB.Exec(`INSERT INTO context_notes (ref, session_id, project_path, label, content, content_hash, token_est)
			VALUES ('ctx_test1', 'sess-a', '/proj', 'plan', 'hello world', 'hash1', 400)`)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/dashboard/weekly-digest?project_id=/proj", nil)
	rr := httptest.NewRecorder()
	handleDashboardWeeklyDigestJSON(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("weekly digest status=%d body=%s", rr.Code, rr.Body.String())
	}
	var digest WeeklyDigest
	if err := json.Unmarshal(rr.Body.Bytes(), &digest); err != nil {
		t.Fatal(err)
	}
	if digest.WindowDays != 7 {
		t.Fatalf("WindowDays=%d", digest.WindowDays)
	}
	if digest.TokensSaved < 4000 {
		t.Fatalf("TokensSaved=%d", digest.TokensSaved)
	}
	if digest.VirtualStored < 800 || digest.VirtualAccessed < 750 {
		t.Fatalf("VC stored=%d accessed=%d", digest.VirtualStored, digest.VirtualAccessed)
	}
	if len(digest.TopTools) == 0 {
		t.Fatal("expected top tools")
	}
	if !digest.Heuristic.HeuristicApproximate {
		t.Fatal("heuristic should be approximate")
	}
	if !digest.EmbedReliability.Available {
		t.Fatal("embed reliability should be available")
	}

	req2 := httptest.NewRequest(http.MethodGet, "/api/dashboard/context-sessions?project_id=/proj", nil)
	rr2 := httptest.NewRecorder()
	handleDashboardContextSessionsJSON(rr2, req2)
	if rr2.Code != http.StatusOK {
		t.Fatalf("context sessions status=%d body=%s", rr2.Code, rr2.Body.String())
	}
	var sessions ContextSessionsResponse
	if err := json.Unmarshal(rr2.Body.Bytes(), &sessions); err != nil {
		t.Fatal(err)
	}
	if sessions.WindowDays != 30 {
		t.Fatalf("WindowDays=%d", sessions.WindowDays)
	}
	if len(sessions.Sessions) != 1 {
		t.Fatalf("sessions=%d", len(sessions.Sessions))
	}
	s0 := sessions.Sessions[0]
	if s0.SessionID != "sess-a" || !s0.FetchedAfterStore {
		t.Fatalf("story=%+v", s0)
	}
	if s0.VirtualTokensStored != 800 || s0.VirtualTokensAccessed != 750 {
		t.Fatalf("tokens store/access=%d/%d", s0.VirtualTokensStored, s0.VirtualTokensAccessed)
	}

	req3 := httptest.NewRequest(http.MethodGet, "/api/dashboard/stats?project_id=/proj", nil)
	rr3 := httptest.NewRecorder()
	handleDashboardStatsJSON(rr3, req3)
	if rr3.Code != http.StatusOK {
		t.Fatalf("stats status=%d body=%s", rr3.Code, rr3.Body.String())
	}
	var stats map[string]interface{}
	if err := json.Unmarshal(rr3.Body.Bytes(), &stats); err != nil {
		t.Fatal(err)
	}
	if stats["HeuristicApproximate"] != true {
		t.Fatalf("HeuristicApproximate missing: %+v", stats)
	}
	if stats["HeuristicLabel"] != "approximate" {
		t.Fatalf("HeuristicLabel=%v", stats["HeuristicLabel"])
	}
	rounds, ok := stats["ApproxRoundsAvoided"].(float64)
	if !ok || rounds <= 0 {
		t.Fatalf("ApproxRoundsAvoided=%v", stats["ApproxRoundsAvoided"])
	}
}
