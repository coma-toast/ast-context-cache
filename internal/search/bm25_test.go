package search

import (
	"strings"
	"testing"
)

func TestQueryTerms(t *testing.T) {
	terms := QueryTerms("tools/list MCP")
	if len(terms) != 2 {
		t.Fatalf("terms=%v", terms)
	}
	q := BuildFTSQuery(terms)
	if q == "" || !strings.Contains(strings.ToLower(q), "tools") || !strings.Contains(strings.ToLower(q), "mcp") {
		t.Fatalf("fts query=%q", q)
	}
}
