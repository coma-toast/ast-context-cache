package mcp

import (
	"strings"
	"testing"
)

func TestContextToolsRegistered(t *testing.T) {
	names := map[string]bool{}
	for _, tdef := range GetTools() {
		names[tdef.Name] = true
	}
	for _, want := range []string{"store_context", "fetch_context", "list_context", "search_context", "flush_context", "edit_context", "define_context_fn", "apply_context_fn", "list_context_fns", "report_kv_repair_event", "store_memory", "recall_memory", "forget_memory", "handoff", "open_handoff", "scratchpad"} {
		if !names[want] {
			t.Fatalf("missing tool %s", want)
		}
	}
}

func TestContextToolTiers(t *testing.T) {
	tierOf := map[string]Tier{}
	for _, tdef := range GetTools() {
		tierOf[tdef.Name] = tdef.Tier
	}
	if tierOf["define_context_fn"] != TierExtended || tierOf["apply_context_fn"] != TierExtended || tierOf["list_context_fns"] != TierCore {
		t.Fatalf("context fn tiers: define=%v apply=%v list=%v", tierOf["define_context_fn"], tierOf["apply_context_fn"], tierOf["list_context_fns"])
	}
	if tierOf["edit_context"] != TierExtended || tierOf["store_context"] != TierExtended || tierOf["flush_context"] != TierExtended || tierOf["report_kv_repair_event"] != TierExtended || tierOf["store_memory"] != TierExtended || tierOf["forget_memory"] != TierExtended {
		t.Fatalf("write tools tier: edit=%v store=%v flush=%v store_mem=%v forget_mem=%v", tierOf["edit_context"], tierOf["store_context"], tierOf["flush_context"], tierOf["store_memory"], tierOf["forget_memory"])
	}
	for _, name := range []string{"fetch_context", "list_context", "search_context", "list_context_fns", "recall_memory", "handoff", "open_handoff", "scratchpad"} {
		if tierOf[name] != TierCore {
			t.Fatalf("%s should be core", name)
		}
	}
}

func TestFilterToolsIncludesContextReadAtCore(t *testing.T) {
	cfg := ServerConfig{ActiveTier: TierCore, CodeMode: false}
	var names []string
	for _, t := range FilterTools(cfg) {
		names = append(names, t.Name)
	}
	joined := strings.Join(names, ",")
	for _, want := range []string{"fetch_context", "list_context", "search_context", "recall_memory", "handoff", "open_handoff", "scratchpad"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("core tier missing %s in %s", want, joined)
		}
	}
	if strings.Contains(joined, "store_context") || strings.Contains(joined, "flush_context") || strings.Contains(joined, "edit_context") {
		t.Fatalf("core tier should not include write context tools: %s", joined)
	}
}

// The flag is the only thing standing between an agent and unrestricted writes to
// its own stored context, so it has to actually gate the tool.
func TestEditContextToolFlagGating(t *testing.T) {
	cfg := ServerConfig{ActiveTier: TierComplete, CodeMode: false}
	if len(FilterTools(cfg)) == 0 {
		t.Fatal("expected tools at complete tier")
	}
	found := false
	for _, td := range GetTools() {
		if td.Name == "edit_context" {
			found = true
		}
	}
	if !found {
		t.Fatal("edit_context missing from the tool catalog")
	}
}
