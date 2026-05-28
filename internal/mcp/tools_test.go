package mcp

import (
	"os"
	"path/filepath"
	"testing"
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
	if err := os.WriteFile(path, []byte("{not json"), 0644); err != nil {
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
	if err := os.WriteFile(path, []byte(`{"index_files":{"enabled":true,"tier":"EXTENDED"}}`), 0644); err != nil {
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
