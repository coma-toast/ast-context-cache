package dashboard

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

func handleDockerModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	rawURL := strings.TrimSpace(r.URL.Query().Get("url"))
	if rawURL == "" {
		rawURL = embedder.DefaultDockerURL
	}
	selected := strings.TrimSpace(r.URL.Query().Get("selected"))
	models, err := embedder.ListDMRModels(rawURL)
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

func loadDockerModels(data *components.SettingsData) {
	if !strings.EqualFold(strings.TrimSpace(data.EmbedBackend), "docker") {
		return
	}
	models, err := embedder.ListDMRModels(data.EmbedDockerURL)
	if err != nil {
		data.EmbedDockerModelsErr = err.Error()
		return
	}
	if data.EmbedDockerModel != "" && !sliceContains(models, data.EmbedDockerModel) {
		models = append([]string{data.EmbedDockerModel}, models...)
	}
	data.EmbedDockerModels = models
}
