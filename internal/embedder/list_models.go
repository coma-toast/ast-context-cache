package embedder

import (
	"fmt"
	"strings"
)

// ListModels returns remote model ids for backends that support discovery.
func ListModels(s Settings) ([]string, error) {
	backend := strings.ToLower(strings.TrimSpace(s.Backend))
	if backend == "" {
		backend = "onnx"
	}
	switch backend {
	case "docker":
		url := strings.TrimSpace(s.DockerURL)
		if url == "" {
			url = DefaultDockerURL
		}
		return ListDMRModels(url)
	case "ollama":
		host := strings.TrimSpace(s.OllamaHost)
		if host != "" && !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
			host = "http://" + host
		}
		if host == "" {
			host = "http://127.0.0.1:11434"
		}
		return ListOllamaModels(host)
	case "openai", "litellm":
		base := strings.TrimSpace(s.OpenAIBaseURL)
		if base == "" {
			base = "https://api.openai.com/v1"
		}
		return ListOpenAIModels(base, s.OpenAIAPIKey)
	default:
		return nil, fmt.Errorf("model listing not available for %q backend", backend)
	}
}

// CurrentModel returns the configured model id for listable backends.
func CurrentModel(s Settings) string {
	switch strings.ToLower(strings.TrimSpace(s.Backend)) {
	case "docker":
		return strings.TrimSpace(s.DockerModel)
	case "ollama":
		return strings.TrimSpace(s.OllamaModel)
	case "openai", "litellm":
		return strings.TrimSpace(s.OpenAIModel)
	default:
		return ""
	}
}
