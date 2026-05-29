package embedder

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestBackendConfigReady(t *testing.T) {
	if !BackendConfigReady("onnx") || !BackendConfigReady("docker") {
		t.Fatal("onnx/docker should always be ready")
	}
	t.Setenv("EMBED_OPENAI_MODEL", "text-embedding-3-small")
	if !BackendConfigReady("openai") {
		t.Fatal("openai with model env should be ready")
	}
}

func TestResolveStartupBackend_envOpenAI(t *testing.T) {
	t.Setenv("EMBED_BACKEND", "openai")
	t.Setenv("EMBED_OPENAI_MODEL", "text-embedding-3-small")
	if got := ResolveStartupBackend(); got != "openai" {
		t.Fatalf("got %q want openai", got)
	}
}

// Regression: dashboard saved EMBED_BACKEND=openai without a model must not prevent ast-mcp start.
func TestResolveStartupBackend_incompleteOpenAI_fallsBackToOnnx(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("EMBED_BACKEND", "")
	t.Setenv("EMBED_OPENAI_MODEL", "")
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if err := db.SetSetting("EMBED_BACKEND", "openai"); err != nil {
		t.Fatal(err)
	}
	if got := ResolveStartupBackend(); got != "onnx" {
		t.Fatalf("got %q want onnx fallback", got)
	}
}
