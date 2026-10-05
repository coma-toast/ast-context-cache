package search

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSearchFilters_MatchesSymbol(t *testing.T) {
	project := "/Users/proj/app"
	f := &SearchFilters{
		PathPrefix: "internal/mcp",
		Kinds:      []string{"function"},
		Language:   "go",
	}
	if !f.MatchesSymbol(project+"/internal/mcp/server.go", "function", project) {
		t.Fatal("expected match for path+kind+go")
	}
	if f.MatchesSymbol(project+"/internal/db/x.go", "function", project) {
		t.Fatal("path should exclude db")
	}
	if f.MatchesSymbol(project+"/internal/mcp/server.go", "type", project) {
		t.Fatal("kind should exclude type")
	}
	if f.MatchesSymbol(project+"/internal/mcp/readme.md", "function", project) {
		t.Fatal("language go should exclude md")
	}
}

func TestFileHasPathPrefix_Relative(t *testing.T) {
	p := "/proj/root"
	file := "/proj/root/pkg/foo.go"
	if !fileHasPathPrefix(file, p, "pkg") {
		t.Fatal("prefix pkg")
	}
	if !fileHasPathPrefix(file, p, "pkg/") {
		t.Fatal("prefix pkg/")
	}
	if fileHasPathPrefix(file, p, "other") {
		t.Fatal("other should not match")
	}
}

// A bare, unbounded HasPrefix used to let path_prefix="internal/mcp" match a
// sibling directory like "internal/mcpextra/...", since the string "internal/
// mcpextra" does start with "internal/mcp". Only an exact match or a
// "/"-bounded subdirectory match should count.
func TestFileHasPathPrefix_DoesNotMatchSiblingDirectory(t *testing.T) {
	p := "/proj/root"
	if !fileHasPathPrefix(p+"/internal/mcp/server.go", p, "internal/mcp") {
		t.Fatal("real subdirectory should match")
	}
	if fileHasPathPrefix(p+"/internal/mcpextra/server.go", p, "internal/mcp") {
		t.Fatal("sibling directory sharing the prefix string should not match")
	}
	if fileHasPathPrefix(p+"/internal/mcpextra/server.go", "/other/root", "/proj/root/internal/mcp") {
		t.Fatal("sibling directory should not match an absolute prefix either")
	}
	if !fileHasPathPrefix(p+"/internal/mcp/server.go", "/other/root", "/proj/root/internal/mcp") {
		t.Fatal("real subdirectory should match an absolute prefix")
	}
}

func TestLanguageExtensions(t *testing.T) {
	exts := languageExtensions("typescript")
	if len(exts) != 2 {
		t.Fatalf("typescript: got %v", exts)
	}
}

func TestParseSearchFilters_Empty(t *testing.T) {
	if ParseSearchFilters(map[string]interface{}{}) != nil {
		t.Fatal("expected nil")
	}
}

func TestParseSearchFilters_KindsString(t *testing.T) {
	f := ParseSearchFilters(map[string]interface{}{
		"kinds": "function, method",
	})
	if f == nil || len(f.Kinds) != 2 {
		t.Fatalf("got %#v", f)
	}
}

func TestParseSearchFilters_DedupeKinds(t *testing.T) {
	f := ParseSearchFilters(map[string]interface{}{
		"kinds": "function, function",
		"kind":  "function",
	})
	if f == nil || len(f.Kinds) != 1 || f.Kinds[0] != "function" {
		t.Fatalf("expected single function, got %#v", f.Kinds)
	}
}

func TestSymbolFilterSQL_KindsAndPath(t *testing.T) {
	f := &SearchFilters{
		Kinds:      []string{"function"},
		PathPrefix: "internal/mcp",
	}
	frag, args := symbolFilterSQL(f, "/proj/root")
	if frag == "" {
		t.Fatal("expected fragment")
	}
	if len(args) != 3 {
		t.Fatalf("args: %v", args)
	}
	if args[0] != "function" {
		t.Fatalf("kind arg: %v", args[0])
	}
	wantExact := filepath.ToSlash(filepath.Join("/proj/root", "internal/mcp"))
	if args[1] != wantExact {
		t.Fatalf("exact path: got %v want %v", args[1], wantExact)
	}
	if args[2] != wantExact+"/%" {
		t.Fatalf("prefix like: %v", args[2])
	}
}

func TestSymbolFilterSQL_LanguageGo(t *testing.T) {
	f := &SearchFilters{Language: "go"}
	frag, args := symbolFilterSQL(f, "/p")
	if !strings.Contains(frag, "LIKE") || len(args) != 1 || args[0] != "%.go" {
		t.Fatalf("got %q %v", frag, args)
	}
}

func TestSearchFiltersNormalizedKey(t *testing.T) {
	t.Parallel()
	const project = "/Users/proj/app"
	tests := []struct {
		name string
		a, b *SearchFilters
	}{
		{"kinds case and order", &SearchFilters{Kinds: []string{"Method", "function"}}, &SearchFilters{Kinds: []string{"function", "method"}}},
		{"kinds duplicates", &SearchFilters{Kinds: []string{"function", "FUNCTION", " function "}}, &SearchFilters{Kinds: []string{"function"}}},
		{"go alias", &SearchFilters{Language: "golang"}, &SearchFilters{Language: "Go"}},
		{"ts alias", &SearchFilters{Language: "ts"}, &SearchFilters{Language: "typescript"}},
		{"yml alias", &SearchFilters{Language: " YML "}, &SearchFilters{Language: "yaml"}},
		{"dot slash prefix", &SearchFilters{PathPrefix: "./internal/mcp"}, &SearchFilters{PathPrefix: "internal/mcp"}},
		{"trailing slash prefix", &SearchFilters{PathPrefix: "internal/mcp/"}, &SearchFilters{PathPrefix: "internal/mcp"}},
		{"absolute prefix", &SearchFilters{PathPrefix: project + "/internal/mcp/"}, &SearchFilters{PathPrefix: "internal/mcp"}},
		{"doubled separators", &SearchFilters{PathPrefix: "internal//mcp"}, &SearchFilters{PathPrefix: "internal/mcp"}},
		{"project root is no filter", &SearchFilters{PathPrefix: project}, nil},
		{"dot is no filter", &SearchFilters{PathPrefix: "./"}, &SearchFilters{}},
		{"all combined", &SearchFilters{PathPrefix: "./internal/", Kinds: []string{"Class", "function"}, Language: "py"}, &SearchFilters{PathPrefix: project + "/internal", Kinds: []string{"function", "class"}, Language: "python"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.b.NormalizedKey(project), tt.a.NormalizedKey(project))
		})
	}
}

func TestSearchFiltersNormalizedKeyDistinguishes(t *testing.T) {
	t.Parallel()
	const project = "/Users/proj/app"
	keys := map[string]*SearchFilters{
		"none":     nil,
		"path":     {PathPrefix: "internal"},
		"subpath":  {PathPrefix: "internal/mcp"},
		"kind":     {Kinds: []string{"function"}},
		"go":       {Language: "go"},
		"python":   {Language: "python"},
		"unknown":  {Language: "cobol"},
		"combined": {PathPrefix: "internal", Language: "go"},
	}
	seen := map[string]string{}
	for name, f := range keys {
		k := f.NormalizedKey(project)
		if prev, ok := seen[k]; ok {
			t.Fatalf("%s and %s share key %q", prev, name, k)
		}
		seen[k] = name
	}
	assert.Equal(t, "", (*SearchFilters)(nil).NormalizedKey(project))
	assert.Equal(t, "p:internal/mcp|k:function|l:go", (&SearchFilters{PathPrefix: "internal/mcp", Kinds: []string{"function"}, Language: "golang"}).NormalizedKey(project))
}
