package dashboard

import (
	"encoding/json"
	"net/http"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

func handleEmbedderRetry(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	res := embedder.NudgeRecovery()
	if !res.OK {
		w.WriteHeader(http.StatusBadGateway)
	}
	json.NewEncoder(w).Encode(res)
}

func handleEmbedderDismissAlert(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	res := embedder.DismissAlert()
	if !res.OK {
		w.WriteHeader(http.StatusConflict)
	}
	json.NewEncoder(w).Encode(res)
}
