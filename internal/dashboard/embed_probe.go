package dashboard

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
)

func handleEmbedderTest(w http.ResponseWriter, r *http.Request) {
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
	modelDir := strings.TrimSpace(req["MODEL_DIR"])
	if modelDir == "" {
		exePath, _ := os.Executable()
		modelDir = filepath.Join(filepath.Dir(exePath), "model")
	}
	res := embedder.TestSettings(embedder.SettingsFromMap(req), modelDir)
	if !res.OK {
		w.WriteHeader(http.StatusBadGateway)
	}
	json.NewEncoder(w).Encode(res)
}

func handleEmbedderVerifyRunning(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	res := embedder.VerifyRunning(mcp.GetEmbedder())
	if !res.OK {
		w.WriteHeader(http.StatusBadGateway)
	}
	json.NewEncoder(w).Encode(res)
}
