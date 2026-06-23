package embedder

import (
	"encoding/json"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const embedProfilesKey = "embed_backend_profiles"

func profileKeysForBackend(backend string) []string {
	switch normalizeEmbedBackend(backend) {
	case "onnx":
		return []string{"MODEL_DIR"}
	case "http":
		return []string{"EMBED_HTTP_URL", "EMBED_HTTP_BEARER"}
	case "ollama":
		return []string{"OLLAMA_HOST", "OLLAMA_EMBED_MODEL"}
	case "openai":
		return []string{"EMBED_OPENAI_BASE_URL", "EMBED_OPENAI_API_KEY", "EMBED_OPENAI_MODEL", "EMBED_OPENAI_DIMENSIONS"}
	case "docker":
		return []string{"EMBED_DOCKER_URL", "EMBED_DOCKER_MODEL", "EMBED_DOCKER_DIMENSIONS"}
	default:
		return nil
	}
}

func loadStoredProfile(backend string) map[string]string {
	raw := strings.TrimSpace(db.GetSetting(embedProfilesKey, ""))
	if raw == "" {
		return nil
	}
	var profiles map[string]map[string]string
	if err := json.Unmarshal([]byte(raw), &profiles); err != nil {
		return nil
	}
	return profiles[normalizeEmbedBackend(backend)]
}

// SettingsForStoredProfile builds embed settings from a saved backend profile with env/DB fallback.
func SettingsForStoredProfile(backend string) Settings {
	backend = normalizeEmbedBackend(backend)
	if backend == "" {
		backend = "onnx"
	}
	m := map[string]string{"EMBED_BACKEND": backend}
	if p := loadStoredProfile(backend); p != nil {
		for k, v := range p {
			m[k] = v
		}
	}
	for _, k := range profileKeysForBackend(backend) {
		if strings.TrimSpace(m[k]) == "" {
			m[k] = EffectiveEnv(k)
		}
	}
	return SettingsFromMap(m)
}
