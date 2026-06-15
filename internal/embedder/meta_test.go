package embedder

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestWiredSnapshotFrozenAfterFreeze(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	SetActive("docker", "ai/qwen3-embedding", 768, "dmr", "http://127.0.0.1:12434/engines/v1/embeddings")
	FreezeWiredSnapshot()
	SetActive("openai", "vm/nomic-embed-text-v1.5", 768, "openai", "http://127.0.0.1:8080/v1/embeddings")
	wb, wm, _, _, _ := WiredSnapshot()
	if wb != "docker" || wm != "ai/qwen3-embedding" {
		t.Fatalf("wired=%s/%s want docker/ai/qwen3-embedding", wb, wm)
	}
	ab, am, _, _, _ := ActiveSnapshot()
	if ab != "openai" || am != "vm/nomic-embed-text-v1.5" {
		t.Fatalf("active=%s/%s", ab, am)
	}
	align := AlignmentStatus()
	if align.ActiveBackend != "docker" || align.ActiveModel != "ai/qwen3-embedding" {
		t.Fatalf("alignment active=%s/%s", align.ActiveBackend, align.ActiveModel)
	}
}

func TestNewFromSettingsDoesNotMutateWiredSnapshot(t *testing.T) {
	SetActive("ollama", "nomic-embed-text", 768, "ollama", "http://127.0.0.1:11434/api/embed")
	FreezeWiredSnapshot()
	_, err := NewFromSettings(Settings{
		Backend:     "openai",
		OpenAIModel: "text-embedding-3-small",
	}, "")
	if err != nil {
		t.Fatalf("NewFromSettings: %v", err)
	}
	wb, _, _, _, _ := WiredSnapshot()
	if wb != "ollama" {
		t.Fatalf("wired backend=%q want ollama", wb)
	}
}
