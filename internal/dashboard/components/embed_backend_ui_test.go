package components

import "testing"

func TestEmbedModelsForBackend(t *testing.T) {
	models := []string{"a", "b"}
	if got := EmbedModelsForBackend("docker", "docker", models); len(got) != 2 {
		t.Fatalf("active panel should get models: %v", got)
	}
	if got := EmbedModelsForBackend("docker", "openai", models); got != nil {
		t.Fatalf("inactive panel should get nil, got %v", got)
	}
}

func TestEmbedModelsErrForBackend(t *testing.T) {
	if got := EmbedModelsErrForBackend("openai", "openai", "boom"); got != "boom" {
		t.Fatalf("got %q", got)
	}
	if got := EmbedModelsErrForBackend("docker", "openai", "boom"); got != "" {
		t.Fatalf("got %q", got)
	}
}
