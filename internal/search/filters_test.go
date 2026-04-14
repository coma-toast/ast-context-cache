package search

import (
	"path/filepath"
	"strings"
	"testing"
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
