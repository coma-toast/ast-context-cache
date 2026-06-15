package dashboard

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

const embedProfilesKey = "embed_backend_profiles"

type embedProfiles map[string]map[string]string

func profileKeysForBackend(backend string) []string {
	switch components.EmbedBackendUI(backend) {
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

func loadEmbedProfiles() embedProfiles {
	raw := strings.TrimSpace(db.GetSetting(embedProfilesKey, ""))
	if raw == "" {
		return embedProfiles{}
	}
	var p embedProfiles
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		return embedProfiles{}
	}
	if p == nil {
		return embedProfiles{}
	}
	return p
}

func saveEmbedProfiles(p embedProfiles) error {
	if p == nil {
		p = embedProfiles{}
	}
	b, err := json.Marshal(p)
	if err != nil {
		return err
	}
	return db.SetSetting(embedProfilesKey, string(b))
}

func snapshotEmbedProfileFromDB(backend string) {
	backend = components.EmbedBackendUI(backend)
	keys := profileKeysForBackend(backend)
	if len(keys) == 0 {
		return
	}
	fields := make(map[string]string, len(keys))
	for _, k := range keys {
		fields[k] = db.GetSetting(k, "")
	}
	profiles := loadEmbedProfiles()
	profiles[backend] = fields
	_ = saveEmbedProfiles(profiles)
}

func restoreEmbedProfile(backend string) {
	backend = components.EmbedBackendUI(backend)
	p, ok := loadEmbedProfiles()[backend]
	if !ok {
		return
	}
	for k, v := range p {
		_ = db.SetSetting(k, v)
	}
}

func normalizeEmbedBackendValue(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "litellm":
		return "openai"
	default:
		return strings.TrimSpace(value)
	}
}

// ApplyBackendDefaultsIfEmpty writes sensible defaults for unset keys of the selected backend.
func ApplyBackendDefaultsIfEmpty(backend string) {
	for k, v := range backendDefaults(backend) {
		if strings.TrimSpace(db.GetSetting(k, "")) == "" {
			_ = db.SetSetting(k, v)
		}
	}
}

func backendDefaults(backend string) map[string]string {
	switch components.EmbedBackendUI(backend) {
	case "docker":
		return map[string]string{
			"EMBED_DOCKER_URL":        embedder.DefaultDockerURL,
			"EMBED_DOCKER_MODEL":      embedder.DefaultDockerModel,
			"EMBED_DOCKER_DIMENSIONS": embedder.DefaultDockerDimensions,
		}
	case "ollama":
		return map[string]string{
			"OLLAMA_HOST":         "http://127.0.0.1:11434",
			"OLLAMA_EMBED_MODEL":  "nomic-embed-text",
		}
	case "openai":
		return map[string]string{
			"EMBED_OPENAI_BASE_URL": "https://api.openai.com/v1",
		}
	case "http":
		return map[string]string{
			"EMBED_HTTP_URL": "http://127.0.0.1:8080/embed",
		}
	default:
		return nil
	}
}

// PersistEmbedSettings saves embed keys, updates the per-backend profile, and applies defaults.
func PersistEmbedSettings(m map[string]string) error {
	oldBackend := components.EmbedBackendUI(db.GetSetting("EMBED_BACKEND", ""))
	newBackend := components.EmbedBackendUI(m["EMBED_BACKEND"])
	if newBackend == "" {
		newBackend = oldBackend
	}
	snapshotEmbedProfileFromDB(oldBackend)
	if newBackend != oldBackend && strings.TrimSpace(m["EMBED_BACKEND"]) != "" {
		if err := db.SetSetting("EMBED_BACKEND", normalizeEmbedBackendValue(m["EMBED_BACKEND"])); err != nil {
			return err
		}
		restoreEmbedProfile(newBackend)
		ApplyBackendDefaultsIfEmpty(newBackend)
		for _, k := range profileKeysForBackend(newBackend) {
			v, ok := m[k]
			if !ok || strings.TrimSpace(v) == "" {
				continue
			}
			if err := db.SetSetting(k, v); err != nil {
				return err
			}
		}
	} else {
		for _, k := range embedder.SettingKeys() {
			v, ok := m[k]
			if !ok {
				continue
			}
			if k == "EMBED_BACKEND" {
				v = normalizeEmbedBackendValue(v)
			}
			if err := db.SetSetting(k, v); err != nil {
				return err
			}
		}
		ApplyBackendDefaultsIfEmpty(newBackend)
	}
	snapshotEmbedProfileFromDB(newBackend)
	return nil
}

func onEmbedSettingChanged(key string) {
	if !embedder.IsSettingKey(key) || key == "EMBED_BACKEND" {
		return
	}
	backend := components.EmbedBackendUI(db.GetSetting("EMBED_BACKEND", ""))
	snapshotEmbedProfileFromDB(backend)
}

func switchEmbedBackend(oldBackend, newBackend, newValue string) error {
	snapshotEmbedProfileFromDB(oldBackend)
	if err := db.SetSetting("EMBED_BACKEND", newValue); err != nil {
		return err
	}
	restoreEmbedProfile(newBackend)
	ApplyBackendDefaultsIfEmpty(newBackend)
	snapshotEmbedProfileFromDB(newBackend)
	return nil
}

func handleEmbedSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	var req map[string]string
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	if err := PersistEmbedSettings(req); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	writeEmbedSettingsOK(w)
}
