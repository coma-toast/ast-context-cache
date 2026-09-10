package search

import (
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
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

func TestBuildTrigramQuery(t *testing.T) {
	if q := BuildTrigramQuery(nil); q != "" {
		t.Fatalf("expected empty query for no terms, got %q", q)
	}
	if q := BuildTrigramQuery([]string{"ab"}); q != "" {
		t.Fatalf("expected terms under 3 chars to be dropped (trigram can't match them), got %q", q)
	}
	q := BuildTrigramQuery([]string{"cache", "http"})
	if !strings.Contains(q, `"cache"`) || !strings.Contains(q, `"http"`) || !strings.Contains(q, " OR ") {
		t.Fatalf("expected quoted terms ORed together, got %q", q)
	}
	if q := BuildTrigramQuery([]string{`weird"quote`}); !strings.Contains(q, `""`) {
		t.Fatalf("expected embedded quote to be escaped as doubled quote, got %q", q)
	}
}

// TestBM25SearchFallsBackToTrigramForSubstringMiss covers the exact gap that motivated
// adding the trigram index: the default unicode61 tokenizer treats "VectorCache" as one
// token, so BuildFTSQuery's prefix query "cache*" never matches it. BM25Search must
// still find it by falling through to TrigramSearch's substring match.
func TestBM25SearchFallsBackToTrigramForSubstringMiss(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(db.Close)

	if _, err := db.IndexDB.Exec(
		`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"VectorCache", "type", "cache.go", 1, 10, "type VectorCache struct{}", "pkg.VectorCache", "/proj",
	); err != nil {
		t.Fatal(err)
	}

	results := BM25Search("cache", "/proj", nil)
	found := false
	for _, r := range results {
		if name, _ := r.Data["name"].(string); name == "VectorCache" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected BM25Search to find VectorCache via trigram substring match, got %+v", results)
	}
}
