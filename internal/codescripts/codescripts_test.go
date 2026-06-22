package codescripts_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
)

func TestResolveBuiltinScript(t *testing.T) {
	code, err := codescripts.ResolveScript("compact-symbol-list", "")
	if err != nil {
		t.Fatal(err)
	}
	if code == "" {
		t.Fatal("expected builtin code")
	}
}

func TestManifestPathJail(t *testing.T) {
	dir := t.TempDir()
	scriptsDir := filepath.Join(dir, "scripts", "code-mode")
	if err := os.MkdirAll(scriptsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scriptsDir, "manifest.json"), []byte(`[{"id":"evil","title":"x","match":{},"code_file":"../outside.js"}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	codescripts.InvalidateRepoCache(dir)
	_, err := codescripts.ResolveScript("evil", dir)
	if err == nil {
		t.Fatal("expected path jail to block escape")
	}
}

func TestRepoOverrideExtends(t *testing.T) {
	dir := t.TempDir()
	scriptsDir := filepath.Join(dir, "scripts", "code-mode")
	if err := os.MkdirAll(scriptsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := `[{"id":"custom-compact","title":"Custom","match":{"min_results":1},"code_file":"custom.js","extends":"compact-symbol-list"}]`
	if err := os.WriteFile(filepath.Join(scriptsDir, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scriptsDir, "custom.js"), []byte(`return [{only:"custom"}];`), 0o644); err != nil {
		t.Fatal(err)
	}
	codescripts.InvalidateRepoCache(dir)
	code, err := codescripts.ResolveScript("custom-compact", dir)
	if err != nil {
		t.Fatal(err)
	}
	if code != `return [{only:"custom"}];` {
		t.Fatalf("got %q", code)
	}
}

func TestMatchHintsMinResults(t *testing.T) {
	rows := make([]map[string]interface{}, 12)
	for i := range rows {
		rows[i] = map[string]interface{}{"name": "fn", "kind": "function", "file": "a.go"}
	}
	hints := codescripts.MatchHints("get_context_capsule", "auth", "", rows)
	if len(hints) == 0 {
		t.Fatal("expected hints for 12 results")
	}
	if hints[0].ScriptID == "" {
		t.Fatal("missing script_id")
	}
}

func TestMatchHintsQueryRegex(t *testing.T) {
	rows := make([]map[string]interface{}, 6)
	for i := range rows {
		rows[i] = map[string]interface{}{"name": "Export", "kind": "function", "file": "b.go"}
	}
	hints := codescripts.MatchHints("search_semantic", "public api surface", "", rows)
	found := false
	for _, h := range hints {
		if h.ScriptID == "exports-only" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected exports-only hint")
	}
}

func TestBuiltinIDs(t *testing.T) {
	ids := codescripts.BuiltinIDs()
	if len(ids) < 5 {
		t.Fatalf("expected builtins, got %v", ids)
	}
}
