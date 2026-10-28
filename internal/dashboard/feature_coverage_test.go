package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/coverage"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
)

func seedCoverageRows(t *testing.T) {
	t.Helper()
	testEmbedDB(t)
	now := time.Now()
	rows := []struct {
		ago                time.Duration
		tool, sid, project string
	}{
		{time.Hour, "get_context_capsule", "sess-a", "/proj"},
		{time.Hour, "store_context", "sess-a", "/proj"},
		{2 * time.Hour, "recall_memory", "sess-b", "/proj"},
		{2 * time.Hour, "file_watcher", "sess-b", "/proj"},
		{3 * time.Hour, "fetch_doc", "sess-c", "/other"},
		{8 * 24 * time.Hour, "get_context_capsule", "sess-old", "/proj"},
	}
	for _, r := range rows {
		ts := now.Add(-r.ago).Format(time.RFC3339)
		if _, err := db.DB.Exec(`INSERT INTO queries (timestamp, tool_name, session_id, project_path) VALUES (?, ?, ?, ?)`, ts, r.tool, r.sid, r.project); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFeatureCoverageAPI(t *testing.T) {
	seedCoverageRows(t)
	tests := []struct {
		name, query                      string
		active, searchless, calls, projs int
		searchPct                        float64
	}{
		{name: "all projects", query: "", active: 3, searchless: 2, calls: 4, projs: 2, searchPct: 33.3},
		{name: "project filter", query: "?project_id=/proj", active: 2, searchless: 1, calls: 3, projs: 1, searchPct: 50},
		{name: "wider window", query: "?days=30&project_id=/proj", active: 3, searchless: 1, calls: 4, projs: 1, searchPct: 66.7},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			handleDashboardFeatureCoverageJSON(rr, httptest.NewRequest(http.MethodGet, "/api/dashboard/feature-coverage"+tc.query, nil))
			if rr.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
			}
			t.Logf("response: %s", rr.Body.String())
			var rep coverage.Report
			if err := json.Unmarshal(rr.Body.Bytes(), &rep); err != nil {
				t.Fatal(err)
			}
			if rep.ActiveSessions != tc.active || rep.SearchlessSessions != tc.searchless || rep.Calls != tc.calls || len(rep.TopProjects) != tc.projs {
				t.Fatalf("report=%+v", rep)
			}
			if rep.Features[0].Feature != coverage.FeatureSearch || rep.Features[0].Pct != tc.searchPct {
				t.Fatalf("search=%+v want pct %v", rep.Features[0], tc.searchPct)
			}
		})
	}
}

func TestWeeklyDigestCoverage(t *testing.T) {
	seedCoverageRows(t)
	rr := httptest.NewRecorder()
	handleDashboardWeeklyDigestJSON(rr, httptest.NewRequest(http.MethodGet, "/api/dashboard/weekly-digest?project_id=/proj", nil))
	var digest WeeklyDigest
	if err := json.Unmarshal(rr.Body.Bytes(), &digest); err != nil {
		t.Fatal(err)
	}
	c := digest.Coverage
	if c == nil || c.ActiveSessions != 2 || c.PrevActiveSessions != 1 || len(c.Features) != len(coverage.Features) {
		t.Fatalf("coverage=%+v", c)
	}
	if s := c.Features[0]; s.Pct != 50 || s.PrevPct != 100 || s.DeltaPct != -50 {
		t.Fatalf("search trend=%+v", s)
	}
}

// TestEveryMCPToolHasFeature keeps coverage.Features in step with the registered tools.
func TestEveryMCPToolHasFeature(t *testing.T) {
	for _, tool := range mcp.GetTools() {
		if coverage.FeatureOf(tool.Name) == "" {
			t.Errorf("tool %s is missing from coverage.Features", tool.Name)
		}
	}
}
