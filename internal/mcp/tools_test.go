package mcp

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/flags"
)

func TestFilterTools_disabledOverride(t *testing.T) {
	cfg := ServerConfig{
		ActiveTier: TierComplete,
		CodeMode:   true,
		ToolConfigs: map[string]*ToolConfig{
			"retrieve": {Enabled: false, Tier: TierCore},
		},
	}
	for _, tool := range FilterTools(cfg) {
		if tool.Name == "retrieve" {
			t.Fatal("retrieve should be filtered out when disabled")
		}
	}
	if IsToolAllowed("retrieve", cfg) {
		t.Fatal("IsToolAllowed should be false when disabled")
	}
}

func TestFilterTools_promoteToCore(t *testing.T) {
	cfg := ServerConfig{
		ActiveTier: TierCore,
		CodeMode:   false,
		ToolConfigs: map[string]*ToolConfig{
			"index_files": {Enabled: true, Tier: TierCore},
		},
	}
	found := false
	for _, tool := range FilterTools(cfg) {
		if tool.Name == "index_files" {
			found = true
		}
	}
	if !found {
		t.Fatal("index_files should appear at core when override tier is core")
	}
}

func TestFilterTools_overrideTierTooHigh(t *testing.T) {
	cfg := ServerConfig{
		ActiveTier: TierCore,
		ToolConfigs: map[string]*ToolConfig{
			"index_files": {Enabled: true, Tier: TierExtended},
		},
	}
	if IsToolAllowed("index_files", cfg) {
		t.Fatal("extended-tier override should not pass core active tier")
	}
}

func TestFilterTools_executeCodeRequiresCodeMode(t *testing.T) {
	cfg := ServerConfig{
		ActiveTier: TierComplete,
		CodeMode:   false,
		ToolConfigs: map[string]*ToolConfig{
			"execute_code": {Enabled: true, Tier: TierComplete},
		},
	}
	if IsToolAllowed("execute_code", cfg) {
		t.Fatal("execute_code should be blocked when CodeMode is false")
	}
	_, reason := toolAccessByName("execute_code", cfg)
	if reason != denyCodeMode {
		t.Fatalf("expected denyCodeMode, got %v", reason)
	}
}

func TestLoadToolConfigs_invalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tools.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AST_MCP_TOOLS_CONFIG", path)
	cfg := LoadToolConfigs()
	if len(cfg) != 0 {
		t.Fatalf("invalid JSON should yield empty config, got %d entries", len(cfg))
	}
}

func TestLoadToolConfigs_normalizesTier(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tools.json")
	if err := os.WriteFile(path, []byte(`{"index_files":{"enabled":true,"tier":"EXTENDED"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AST_MCP_TOOLS_CONFIG", path)
	cfg := LoadToolConfigs()
	c := cfg["index_files"]
	if c == nil || c.Tier != TierExtended {
		t.Fatalf("expected TierExtended, got %#v", c)
	}
}

func TestFilterTools_customDescription(t *testing.T) {
	cfg := ServerConfig{
		ActiveTier: TierCore,
		ToolConfigs: map[string]*ToolConfig{
			"retrieve": {Enabled: true, Tier: TierCore, Description: "Custom retrieve"},
		},
	}
	for _, tool := range FilterTools(cfg) {
		if tool.Name == "retrieve" && tool.Description != "Custom retrieve" {
			t.Fatalf("description = %q", tool.Description)
		}
	}
}

func TestToolDenyMessage(t *testing.T) {
	cfg := ServerConfig{ActiveTier: TierCore}
	if msg := ToolDenyMessage("index_files", cfg, denyTier); msg == "" {
		t.Fatal("expected non-empty deny message")
	}
	if msg := ToolDenyMessage("nope", cfg, denyUnknown); msg != "unknown tool: nope" {
		t.Fatalf("got %q", msg)
	}
}

// setFlagEnvs sets feature-flag env vars for the test and reloads the registry. The Reload
// cleanup is registered before t.Setenv so it runs after the env vars are restored.
func setFlagEnvs(t *testing.T, kv map[string]string) {
	t.Helper()
	t.Cleanup(flags.Reload)
	for k, v := range kv {
		t.Setenv(k, v)
	}
	flags.Reload()
}

// The handoff tools are not registered yet, so these tests gate a scratchpad Tool literal.
func TestToolAccessFeatureFlags(t *testing.T) {
	scratchpad := Tool{Name: "scratchpad", Tier: TierExtended}
	tests := []struct {
		name       string
		master     string
		child      string
		active     Tier
		configs    map[string]*ToolConfig
		wantOK     bool
		wantReason toolDenyReason
	}{
		{name: "flags on", master: "true", child: "true", active: TierExtended, wantOK: true, wantReason: denyNone},
		{name: "child flag off hides tool", master: "true", child: "false", active: TierComplete, wantReason: denyFlag},
		{name: "master flag off hides tool", master: "false", child: "true", active: TierComplete, wantReason: denyFlag},
		{name: "tools.json cannot override flag", master: "true", child: "false", active: TierComplete, configs: map[string]*ToolConfig{"scratchpad": {Enabled: true, Tier: TierCore}}, wantReason: denyFlag},
		{name: "tools.json disables with flag on", master: "true", child: "true", active: TierComplete, configs: map[string]*ToolConfig{"scratchpad": {Enabled: false}}, wantReason: denyDisabled},
		{name: "tier still applies with flag on", master: "true", child: "true", active: TierCore, wantReason: denyTier},
		{name: "tools.json promotes tier with flag on", master: "true", child: "true", active: TierCore, configs: map[string]*ToolConfig{"scratchpad": {Enabled: true, Tier: TierCore}}, wantOK: true, wantReason: denyNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": tt.master, "AST_FEATURE_HANDOFF_SCRATCHPAD": tt.child})
			ok, reason := toolAccess(scratchpad, ServerConfig{ActiveTier: tt.active, ToolConfigs: tt.configs})
			assert.Equal(t, tt.wantOK, ok)
			assert.Equal(t, tt.wantReason, reason)
		})
	}
}

func TestFilterToolsHidesFlagDisabledTools(t *testing.T) {
	all := []Tool{{Name: "scratchpad", Tier: TierCore}, {Name: "get_context_capsule", Tier: TierCore}}
	cfg := ServerConfig{ActiveTier: TierComplete, ToolConfigs: map[string]*ToolConfig{"scratchpad": {Enabled: true, Description: "Custom scratchpad"}}}
	names := func() []string {
		var out []string
		for _, tool := range filterTools(all, cfg) {
			out = append(out, tool.Name)
		}
		return out
	}
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": "true", "AST_FEATURE_HANDOFF_SCRATCHPAD": "false"})
	assert.Equal(t, []string{"get_context_capsule"}, names())
	t.Setenv("AST_FEATURE_HANDOFF_SCRATCHPAD", "true")
	flags.Reload()
	visible := filterTools(all, cfg)
	require.Len(t, visible, 2)
	assert.Equal(t, "Custom scratchpad", visible[0].Description, "tools.json overrides still apply to flag-allowed tools")
}

func TestToolDenyMessageFeatureDisabled(t *testing.T) {
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": "true", "AST_FEATURE_HANDOFF_SCRATCHPAD": "false"})
	msg := ToolDenyMessage("scratchpad", ServerConfig{ActiveTier: TierComplete}, denyFlag)
	assert.Contains(t, msg, "feature_disabled")
	assert.Contains(t, msg, "scratchpad")
	assert.Contains(t, msg, flags.KeyHandoffScratchpad)
}
