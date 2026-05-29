package components

import "testing"

func TestEmbedBackendUI(t *testing.T) {
	tests := []struct{ in, want string }{
		{"", "onnx"},
		{"onnx", "onnx"},
		{"ONNX", "onnx"},
		{"litellm", "openai"},
		{"openai", "openai"},
		{"docker", "docker"},
		{"ollama", "ollama"},
		{"http", "http"},
	}
	for _, tc := range tests {
		if got := EmbedBackendUI(tc.in); got != tc.want {
			t.Errorf("EmbedBackendUI(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
