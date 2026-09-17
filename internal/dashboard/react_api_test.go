package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

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
		Tier          string                                       `json:"tier"`
		CodeMode      bool                                         `json:"code_mode"`
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
