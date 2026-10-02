package mcp

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
)

// Field report #11: search_docs returned unrelated sections (RRF scores ~0.016–0.03) with
// no way to tell them from real hits. Below the floor it must say no_match instead.
func TestSearchDocsReportsNoMatchInsteadOfJunk(t *testing.T) {
	origCfg := srvCfg
	srvCfg = DefaultConfig()
	t.Cleanup(func() { srvCfg = origCfg })
	// DB and HOME come from TestMain; clear doc rows so -count=N runs stay independent.
	clearDocs := func() {
		db.ContextDB.Exec(`DELETE FROM doc_content`)
		db.ContextDB.Exec(`DELETE FROM doc_sources`)
	}
	clearDocs()
	t.Cleanup(clearDocs)
	id, err := docs.AddSource("tailscale-api", "markdown", "https://example.com/tailscale", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.ContextDB.Exec(`INSERT INTO doc_content (source_id, title, content) VALUES (?, ?, ?)`,
		id, "Tailscale API: devices", "List devices in the tailnet. Each device has an attribute for its hostname. The API module returns JSON."); err != nil {
		t.Fatal(err)
	}

	out, isErr := callTool(t, "search_docs", map[string]interface{}{"query": "module __getattr__ PEP 562 lazy attribute"})
	if isErr {
		t.Fatalf("no match is not an error: %v", out)
	}
	if out["no_match"] != true || out["total"] != float64(0) || out["hint"] == nil {
		t.Fatalf("want explicit no_match with hint, got %v", out)
	}
	if out["below_floor"] != float64(1) {
		t.Fatalf("below_floor=%v want 1", out["below_floor"])
	}

	out, _ = callTool(t, "search_docs", map[string]interface{}{"query": "tailscale devices"})
	if out["no_match"] != false || out["total"] != float64(1) {
		t.Fatalf("real match: want 1 result and no_match=false, got %v", out)
	}
	if _, ok := out["hint"]; ok {
		t.Fatalf("hint only belongs on no_match: %v", out)
	}
}
