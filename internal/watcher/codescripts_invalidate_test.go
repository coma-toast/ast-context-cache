package watcher

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
	"github.com/fsnotify/fsnotify"
)

// A repo's own scripts/code-mode/ files were cached on first use with no
// invalidation wiring outside tests — editing a repo's code-mode script
// needed a full ast-mcp restart to take effect. handleFSEvent now invalidates
// the cache the moment a file under that directory changes.
func TestHandleFSEventInvalidatesRepoScriptCache(t *testing.T) {
	dir := t.TempDir()
	scriptsDir := filepath.Join(dir, "scripts", "code-mode")
	if err := os.MkdirAll(scriptsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := `[{"id":"v","title":"V","match":{},"code_file":"v.js"}]`
	if err := os.WriteFile(filepath.Join(scriptsDir, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(scriptsDir, "v.js")
	if err := os.WriteFile(scriptPath, []byte(`return "v1";`), 0o644); err != nil {
		t.Fatal(err)
	}
	codescripts.InvalidateRepoCache(dir)

	code, err := codescripts.ResolveScript("v", dir)
	if err != nil {
		t.Fatal(err)
	}
	if code != `return "v1";` {
		t.Fatalf("got %q, want v1", code)
	}

	// Edit the script on disk, then simulate the watcher observing that change
	// — without the fix, ResolveScript below would still return the stale,
	// cached v1 content.
	if err := os.WriteFile(scriptPath, []byte(`return "v2";`), 0o644); err != nil {
		t.Fatal(err)
	}
	handleFSEvent(fsnotify.Event{Name: scriptPath, Op: fsnotify.Write}, dir, nil)

	code, err = codescripts.ResolveScript("v", dir)
	if err != nil {
		t.Fatal(err)
	}
	if code != `return "v2";` {
		t.Fatalf("got %q, want v2 (cache should have been invalidated)", code)
	}
}
