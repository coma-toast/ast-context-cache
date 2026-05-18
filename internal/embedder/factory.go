package embedder

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// NewForMain selects an embedder from the environment. Must match vector index dimension 768
// (see search.VectorDims); otherwise indexing and search will be inconsistent.
//
// Env:
//
//	EMBED_BACKEND: onnx (default) | http | ollama
//	EMBED_HTTP_URL: default http://127.0.0.1:8080/embed (http backend)
//	EMBED_HTTP_BEARER: optional Bearer token
//	OLLAMA_HOST: default http://127.0.0.1:11434 (ollama backend; same as ollama CLI)
//	OLLAMA_EMBED_MODEL: default nomic-embed-text
func NewForMain(modelDir string) (e Interface, isLoaded func() bool, err error) {
	backend := strings.ToLower(strings.TrimSpace(os.Getenv("EMBED_BACKEND")))
	if backend == "" {
		backend = "onnx"
	}
	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)
	if modelDir == "" {
		modelDir = filepath.Join(exeDir, "model")
	}

	switch backend {
	case "onnx":
		SetActive("onnx", ModelName, Dimensions, "onnxruntime", "")
		if eerr := EnsureModel(modelDir); eerr != nil {
			log.Printf("WARNING: Could not ensure embedder model files: %v", eerr)
		}
		le := NewLazy(modelDir)
		return le, func() bool { return le.IsLoaded() }, nil

	case "http":
		u := strings.TrimSpace(os.Getenv("EMBED_HTTP_URL"))
		if u == "" {
			u = "http://127.0.0.1:8080/embed"
		}
		SetActive("http", "remote", Dimensions, "http", u)
		h := NewHTTPEmbedder(u, os.Getenv("EMBED_HTTP_BEARER"))
		return h, func() bool { return true }, nil

	case "ollama":
		url := os.Getenv("OLLAMA_HOST")
		// ollama CLI on macOS often uses OLLAMA_HOST without scheme
		if url != "" && !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			url = "http://" + url
		}
		if url == "" {
			url = "http://127.0.0.1:11434"
		}
		model := strings.TrimSpace(os.Getenv("OLLAMA_EMBED_MODEL"))
		if model == "" {
			model = "nomic-embed-text"
		}
		SetActive("ollama", model, Dimensions, "ollama", url+"/api/embed")
		oe := NewOllamaEmbedder(url, model)
		return oe, func() bool { return true }, nil

	default:
		return nil, nil, fmt.Errorf("EMBED_BACKEND: unknown %q (use onnx, http, or ollama)", backend)
	}
}
