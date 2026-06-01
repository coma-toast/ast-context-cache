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
	for _, want := range []string{"store_context", "fetch_context", "list_context", "search_context", "flush_context"} {
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
	if tierOf["store_context"] != TierExtended || tierOf["flush_context"] != TierExtended {
		t.Fatalf("write tools tier: store=%v flush=%v", tierOf["store_context"], tierOf["flush_context"])
	}
	for _, name := range []string{"fetch_context", "list_context", "search_context"} {
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
	for _, want := range []string{"fetch_context", "list_context", "search_context"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("core tier missing %s in %s", want, joined)
		}
	}
	if strings.Contains(joined, "store_context") || strings.Contains(joined, "flush_context") {
		t.Fatalf("core tier should not include write context tools: %s", joined)
	}
}
