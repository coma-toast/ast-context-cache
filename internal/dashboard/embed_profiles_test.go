package dashboard

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
)

func TestProfileKeysForBackend(t *testing.T) {
	if len(profileKeysForBackend("docker")) != 3 {
		t.Fatal("docker keys")
	}
	if len(profileKeysForBackend("litellm")) != len(profileKeysForBackend("openai")) {
		t.Fatal("litellm maps to openai keys")
	}
	if components.EmbedBackendUI("litellm") != "openai" {
		t.Fatal("litellm ui")
	}
}

func TestNormalizeEmbedBackendValue(t *testing.T) {
	if normalizeEmbedBackendValue("litellm") != "openai" {
		t.Fatal("litellm normalize")
	}
}
