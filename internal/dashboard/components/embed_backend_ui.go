package components

import "strings"

// EmbedBackendUI maps stored EMBED_BACKEND values to the settings dropdown id.
func EmbedBackendUI(backend string) string {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "", "onnx":
		return "onnx"
	case "litellm":
		return "openai"
	default:
		return strings.ToLower(strings.TrimSpace(backend))
	}
}
