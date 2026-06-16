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

// EmbedModelsForBackend returns discovered models only for the active backend panel.
func EmbedModelsForBackend(activeBackend, panelBackend string, models []string) []string {
	if EmbedBackendUI(activeBackend) == panelBackend {
		return models
	}
	return nil
}

// EmbedModelsErrForBackend returns a list error only for the active backend panel.
func EmbedModelsErrForBackend(activeBackend, panelBackend, err string) string {
	if EmbedBackendUI(activeBackend) == panelBackend {
		return err
	}
	return ""
}
