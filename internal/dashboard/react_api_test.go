package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
