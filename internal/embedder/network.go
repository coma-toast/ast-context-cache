package embedder

import "strings"

// IsNetworkBackend reports backends that call out to a remote or containerized embed API
// (always available at process start; no lazy ONNX load).
func IsNetworkBackend(backend string) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "http", "ollama", "openai", "litellm", "docker":
		return true
	default:
		return false
	}
}
