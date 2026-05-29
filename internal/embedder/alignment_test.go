package embedder

import "testing"

func TestEmbedSnapshotsMatch(t *testing.T) {
	tests := []struct {
		ab, am, cb, cm string
		want           bool
	}{
		{"onnx", ModelName, "onnx", ModelName, true},
		{"onnx", ModelName, "", ModelName, true},
		{"openai", "text-embedding-3-small", "litellm", "text-embedding-3-small", true},
		{"docker", "ai/qwen3-embedding", "docker", "ai/qwen3-embedding", true},
		{"onnx", ModelName, "docker", "ai/qwen3-embedding", false},
		{"openai", "m1", "openai", "m2", false},
		{"openai", "m", "openai", "(unset)", false},
	}
	for _, tc := range tests {
		if got := embedSnapshotsMatch(tc.ab, tc.am, tc.cb, tc.cm); got != tc.want {
			t.Errorf("embedSnapshotsMatch(%q,%q,%q,%q) = %v, want %v", tc.ab, tc.am, tc.cb, tc.cm, got, tc.want)
		}
	}
}

func TestSnapshotForSettings(t *testing.T) {
	b, m, rt, ep, _ := SnapshotForSettings(Settings{Backend: "ollama", OllamaModel: "nomic-embed-text"})
	if b != "ollama" || m != "nomic-embed-text" || rt != "ollama" || ep != "http://127.0.0.1:11434/api/embed" {
		t.Fatalf("ollama snapshot: %s %s %s %s", b, m, rt, ep)
	}
	b, m, _, _, _ = SnapshotForSettings(Settings{Backend: "litellm", OpenAIModel: "embed-model"})
	if b != "openai" || m != "embed-model" {
		t.Fatalf("litellm -> openai: %s %s", b, m)
	}
}
