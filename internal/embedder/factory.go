package embedder

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// NewForMain selects an embedder from the environment. Must match vector index dimension 768
// (see search.VectorDims); otherwise indexing and search will be inconsistent.
//
// Env:
//
//	EMBED_BACKEND: onnx (default) | http | ollama | openai (alias: litellm) | docker
//	EMBED_HTTP_URL: default http://127.0.0.1:8080/embed (http backend)
//	EMBED_HTTP_BEARER: optional Bearer token
//	OLLAMA_HOST: default http://127.0.0.1:11434 (ollama backend; same as ollama CLI)
//	OLLAMA_EMBED_MODEL: default nomic-embed-text
//	EMBED_DOCKER_URL: optional DMR OpenAI API root (default http://127.0.0.1:12434/engines/v1)
//	EMBED_DOCKER_MODEL: default ai/qwen3-embedding
//	EMBED_DOCKER_DIMENSIONS: optional; unset sends 768 in JSON; "0" omits the field
//	EMBED_OPENAI_BASE_URL: OpenAI-compatible API root including /v1 (default https://api.openai.com/v1)
//	EMBED_OPENAI_API_KEY: Bearer token (optional for open local gateways)
//	EMBED_OPENAI_MODEL: required for openai backend (e.g. openai/text-embedding-3-small)
//	EMBED_OPENAI_DIMENSIONS: optional; unset sends 768 in JSON for v3 shortening; "0" omits the field
//	EMBED_REMOTE_TIMEOUT: optional HTTP client timeout for remote backends (default 120s, e.g. 15s)
//
// Dashboard / SQLite: when an env var is unset or empty, NewForMain reads the same key from
// settings (see README). Non-empty env always overrides DB.
func NewForMain(modelDir string) (e Interface, isLoaded func() bool, err error) {
	backend := ResolveStartupBackend()
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
		u := strings.TrimSpace(EffectiveEnv("EMBED_HTTP_URL"))
		if u == "" {
			u = "http://127.0.0.1:8080/embed"
		}
		SetActive("http", "remote", Dimensions, "http", u)
		h := NewHTTPEmbedder(u, EffectiveEnv("EMBED_HTTP_BEARER"))
		return h, func() bool { return true }, nil

	case "ollama":
		url := EffectiveEnv("OLLAMA_HOST")
		// ollama CLI on macOS often uses OLLAMA_HOST without scheme
		if url != "" && !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			url = "http://" + url
		}
		if url == "" {
			url = "http://127.0.0.1:11434"
		}
		model := strings.TrimSpace(EffectiveEnv("OLLAMA_EMBED_MODEL"))
		if model == "" {
			model = "nomic-embed-text"
		}
		SetActive("ollama", model, Dimensions, "ollama", url+"/api/embed")
		oe := NewOllamaEmbedder(url, model)
		return oe, func() bool { return true }, nil

	case "openai", "litellm":
		base := strings.TrimSpace(EffectiveEnv("EMBED_OPENAI_BASE_URL"))
		if base == "" {
			base = "https://api.openai.com/v1"
		}
		model := strings.TrimSpace(EffectiveEnv("EMBED_OPENAI_MODEL"))
		if model == "" {
			return nil, nil, fmt.Errorf("EMBED_OPENAI_MODEL is required when EMBED_BACKEND=openai")
		}
		jsonDims, err := resolveOpenAIDimensions()
		if err != nil {
			return nil, nil, err
		}
		openEmb := NewOpenAIEmbedder(base, EffectiveEnv("EMBED_OPENAI_API_KEY"), model, jsonDims)
		ep := strings.TrimRight(strings.TrimSpace(base), "/") + "/embeddings"
		SetActive("openai", model, Dimensions, "openai", ep)
		return openEmb, func() bool { return true }, nil

	case "docker":
		return newDockerEmbedder()

	default:
		return nil, nil, fmt.Errorf("EMBED_BACKEND: unknown %q (use onnx, http, ollama, openai, or docker)", backend)
	}
}

// resolveOpenAIDimensions mirrors prior env-only rules, with DB fallback when the OS env var is absent.
func resolveOpenAIDimensions() (int, error) {
	if _, ok := os.LookupEnv("EMBED_OPENAI_DIMENSIONS"); ok {
		v := strings.TrimSpace(os.Getenv("EMBED_OPENAI_DIMENSIONS"))
		return parseOpenAIDimensionsValue(v)
	}
	v := strings.TrimSpace(db.GetSetting("EMBED_OPENAI_DIMENSIONS", ""))
	if v == "" {
		return 768, nil
	}
	return parseOpenAIDimensionsValue(v)
}

func parseOpenAIDimensionsValue(v string) (int, error) {
	if v == "" || v == "0" {
		return 0, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("EMBED_OPENAI_DIMENSIONS: %w", err)
	}
	if n <= 0 {
		return 0, nil
	}
	return n, nil
}
