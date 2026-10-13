package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
)

// handleDashboardMCPTierJSON used to re-derive the tier from os.Getenv itself,
// defaulting an unset AST_MCP_TIER to "extended" — but the MCP server's own
// real default (mcp.DefaultConfig) is TierComplete, so this endpoint reported
// the wrong tier whenever the env var was unset. It now reads the server's
// actual live config instead of guessing.
func TestHandleDashboardMCPTierJSONReportsRealConfig(t *testing.T) {
	orig := mcp.GetConfig()
	t.Cleanup(func() { mcp.SetConfig(orig) })

	mcp.SetConfig(mcp.ServerConfig{
		ActiveTier: mcp.TierComplete,
		CodeMode:   true,
		ToolConfigs: map[string]*mcp.ToolConfig{
			"execute_code": {Enabled: false, Tier: mcp.TierCore},
		},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/dashboard/mcp-tier", nil)
	rec := httptest.NewRecorder()
	handleDashboardMCPTierJSON(rec, req)

	var out struct {
		Tier          string `json:"tier"`
		CodeMode      bool   `json:"code_mode"`
		ToolOverrides map[string]struct {
			Enabled bool   `json:"enabled"`
			Tier    string `json:"tier"`
		} `json:"tool_overrides"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v (body=%s)", err, rec.Body.String())
	}
	if out.Tier != "complete" {
		t.Fatalf("tier=%q want complete", out.Tier)
	}
	if !out.CodeMode {
		t.Fatal("code_mode should be true")
	}
	ov, ok := out.ToolOverrides["execute_code"]
	if !ok || ov.Enabled || ov.Tier != "core" {
		t.Fatalf("tool_overrides[execute_code]=%+v want {enabled:false tier:core}", ov)
	}
}

// BF-3: the mcp-tier view reports the tools.json the server loads, so a custom
// AST_MCP_TOOLS_CONFIG is shown instead of the hard-coded ~/.astcache/tools.json.
func TestHandleDashboardMCPTierJSONToolsConfigPath(t *testing.T) {
	custom := filepath.Join(t.TempDir(), "custom-tools.json")
	tests := []struct {
		name       string
		env        string
		write      bool
		wantPath   string
		wantExists bool
	}{
		{name: "env path exists", env: custom, write: true, wantPath: custom, wantExists: true},
		{name: "env path missing", env: custom + ".missing", wantPath: custom + ".missing"},
		{name: "default path", env: "", wantPath: filepath.Join(t.TempDir(), ".astcache", "tools.json")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.env == "" {
				home := filepath.Dir(filepath.Dir(tt.wantPath))
				t.Setenv("HOME", home)
			}
			t.Setenv("AST_MCP_TOOLS_CONFIG", tt.env)
			if tt.write {
				require.NoError(t, os.WriteFile(tt.env, []byte("{}"), 0o644))
			}
			rec := httptest.NewRecorder()
			handleDashboardMCPTierJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/mcp-tier", nil))
			var out struct {
				Path   string `json:"tools_json_path"`
				Exists bool   `json:"tools_json_exists"`
			}
			require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
			assert.Equal(t, tt.wantPath, out.Path)
			assert.Equal(t, tt.wantExists, out.Exists)
		})
	}
}

// BF-10: Tokens saved counts compression + dedup from search/read tools only;
// virtual context writes (store_context) are reported separately.
func TestDashboardStatsExcludesVirtual(t *testing.T) {
	testEmbedDB(t)
	now := time.Now().UTC().Format(time.RFC3339)
	const insertQuery = `INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, dedup_tokens_saved, savings_vs_files, duration_ms)
		VALUES (?, ?, 'sess-v', '/proj-v', ?, ?, ?, 1)`
	_, err := db.DB.Exec(insertQuery, now, "get_context_capsule", 1200, 100, 300)
	require.NoError(t, err)
	_, err = db.DB.Exec(insertQuery, now, "store_context", 9000, 900, 9000)
	require.NoError(t, err)
	rec := httptest.NewRecorder()
	handleDashboardStatsJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/stats?project_id=/proj-v", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var out struct {
		TotalQueries     int
		TokensSaved      int
		DedupTokensSaved int
		SavingsVsFiles   int
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	assert.Equal(t, 2, out.TotalQueries)
	assert.Equal(t, 1200, out.TokensSaved)
	assert.Equal(t, 100, out.DedupTokensSaved)
	assert.Equal(t, 300, out.SavingsVsFiles)
}

func TestDashboardStatsLedgers(t *testing.T) {
	testEmbedDB(t)
	now := time.Now().UTC().Format(time.RFC3339)
	const insertQuery = `INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, dedup_tokens_saved,
		tokens_used, conservative_baseline_tokens, ledger, estimate_method, duration_ms)
		VALUES (?, ?, 'sess-l', '/proj-l', ?, ?, ?, ?, ?, ?, 1)`
	for _, r := range []struct {
		tool                             string
		saved, dedup, used, conservative int
		ledger, method                   string
	}{
		{"get_context_capsule", 1200, 200, 300, 700, "compression", "o200k_base"},
		{"search_semantic", 500, 0, 100, 0, "", "bytes4"},
		{"store_context", 9000, 0, 0, 0, "virtual", "o200k_base"},
		{"store_memory", 40, 0, 0, 0, "virtual", "o200k_base"},
		{"fetch_context", 0, 0, 800, 0, "virtual", "o200k_base"},
		{"recall_memory", 0, 0, 60, 0, "virtual", "o200k_base"},
	} {
		_, err := db.DB.Exec(insertQuery, now, r.tool, r.saved, r.dedup, r.used, r.conservative, r.ledger, r.method)
		require.NoError(t, err)
	}
	rec := httptest.NewRecorder()
	handleDashboardStatsJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/stats?project_id=/proj-l", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var out struct {
		TokensSaved           int
		CompressionSaved      int
		DedupSaved            int
		ConservativeSaved     int
		VirtualStoredTokens   int
		VirtualFetchedTokens  int
		VirtualRecalledTokens int
		EstimatedRows         int
		BaselineDefinitions   map[string]string
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	assert.Equal(t, 1700, out.TokensSaved)
	assert.Equal(t, 1500, out.CompressionSaved, "the legacy ledger-less search row counts as compression")
	assert.Equal(t, 200, out.DedupSaved)
	assert.Equal(t, out.TokensSaved, out.CompressionSaved+out.DedupSaved)
	assert.Equal(t, 400, out.ConservativeSaved)
	assert.Equal(t, 9040, out.VirtualStoredTokens)
	assert.Equal(t, 800, out.VirtualFetchedTokens)
	assert.Equal(t, 60, out.VirtualRecalledTokens)
	assert.Equal(t, 1, out.EstimatedRows)
	assert.Contains(t, out.BaselineDefinitions, "conservative")
	digest := buildWeeklyDigest("/proj-l")
	assert.Equal(t, 1500, digest.CompressionSaved)
	assert.Equal(t, 400, digest.ConservativeSaved)
	assert.Equal(t, 800, digest.VirtualFetchedTokens)
}
