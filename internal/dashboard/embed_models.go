package dashboard

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

func handleEmbedModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	var req map[string]string
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(map[string]any{"error": err.Error(), "models": []string{}})
		return
	}
	s := embedder.SettingsFromMap(req)
	selected := embedder.CurrentModel(s)
	models, err := embedder.ListModels(s)
	if err != nil {
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(map[string]any{"error": err.Error(), "models": []string{}, "selected": ""})
		return
	}
	if selected != "" && !sliceContains(models, selected) {
		selected = ""
	}
	json.NewEncoder(w).Encode(map[string]any{"models": models, "selected": selected})
}

// handleDockerModels is deprecated; forwards to handleEmbedModels for older clients.
func handleDockerModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handleEmbedModels(w, r)
		return
	}
	rawURL := strings.TrimSpace(r.URL.Query().Get("url"))
	if rawURL == "" {
		rawURL = embedder.DefaultDockerURL
	}
	selected := strings.TrimSpace(r.URL.Query().Get("selected"))
	s := embedder.Settings{Backend: "docker", DockerURL: rawURL, DockerModel: selected}
	models, err := embedder.ListModels(s)
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(map[string]any{"error": err.Error(), "models": []string{}})
		return
	}
	if selected != "" && !sliceContains(models, selected) {
		models = append([]string{selected}, models...)
	}
	json.NewEncoder(w).Encode(map[string]any{"models": models, "selected": selected})
}

func sliceContains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}

func loadEmbedModels(data *components.SettingsData) {
	s := embedder.Settings{
		Backend:       data.EmbedBackend,
		OllamaHost:    data.EmbedOllamaHost,
		OllamaModel:   data.EmbedOllamaModel,
		OpenAIBaseURL: data.EmbedOpenAIBaseURL,
		OpenAIAPIKey:  data.EmbedOpenAIAPIKey,
		OpenAIModel:   data.EmbedOpenAIModel,
		DockerURL:     data.EmbedDockerURL,
		DockerModel:   data.EmbedDockerModel,
	}
	current := embedder.CurrentModel(s)
	switch components.EmbedBackendUI(data.EmbedBackend) {
	case "docker", "ollama", "openai":
	default:
		return
	}
	models, err := embedder.ListModels(s)
	if err != nil {
		data.EmbedModelsErr = err.Error()
		return
	}
	if current != "" && !sliceContains(models, current) {
		clearEmbedModelField(data, data.EmbedBackend)
	}
	data.EmbedModels = models
}

func clearEmbedModelField(data *components.SettingsData, backend string) {
	switch components.EmbedBackendUI(backend) {
	case "docker":
		data.EmbedDockerModel = ""
	case "ollama":
		data.EmbedOllamaModel = ""
	case "openai":
		data.EmbedOpenAIModel = ""
	}
}
