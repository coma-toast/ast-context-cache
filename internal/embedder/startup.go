package embedder

import (
	"log"
	"strings"
)

// BackendConfigReady reports whether EffectiveSettings are sufficient to start the requested backend.
func BackendConfigReady(backend string) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "", "onnx", "http", "ollama", "docker":
		return true
	case "openai", "litellm":
		return strings.TrimSpace(EffectiveEnv("EMBED_OPENAI_MODEL")) != ""
	default:
		return false
	}
}

// ResolveStartupBackend picks the embedder to wire at process start. Incomplete remote configs
// fall back to onnx so ast-mcp still starts; dashboard alignment shows the configured vs running gap.
func ResolveStartupBackend() string {
	requested := strings.ToLower(strings.TrimSpace(EffectiveEnv("EMBED_BACKEND")))
	if requested == "" {
		return "onnx"
	}
	if BackendConfigReady(requested) {
		return requested
	}
	log.Printf("WARNING: EMBED_BACKEND=%q is not fully configured (%s); using onnx until embed settings are completed in dashboard",
		requested, backendMissingHint(requested))
	return "onnx"
}

func backendMissingHint(backend string) string {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "openai", "litellm":
		return "set EMBED_OPENAI_MODEL"
	default:
		return "check embed settings"
	}
}
