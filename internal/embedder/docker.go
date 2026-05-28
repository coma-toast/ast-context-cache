package embedder

import (
	"log"
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Docker embed backend uses Docker Model Runner (DMR) OpenAI-compatible embeddings API.
//
// Env:
//
//	EMBED_DOCKER_URL: host or API root (default http://127.0.0.1:12434 → …/engines/v1)
//	EMBED_DOCKER_MODEL: model id (default ai/qwen3-embedding)
//	EMBED_DOCKER_DIMENSIONS: optional; unset sends 768 in JSON; "0" omits the field
func newDockerEmbedder() (Interface, func() bool, error) {
	if p := strings.TrimSpace(EffectiveEnv("EMBED_DOCKER_PROVIDER")); p != "" {
		log.Printf("WARNING: EMBED_DOCKER_PROVIDER=%q is ignored; docker backend uses Docker Model Runner (port 12434). Use EMBED_BACKEND=ollama or http for other servers.", p)
	}
	base := normalizeDMRBase(strings.TrimSpace(EffectiveEnv("EMBED_DOCKER_URL")))
	model := strings.TrimSpace(EffectiveEnv("EMBED_DOCKER_MODEL"))
	if model == "" {
		model = "ai/qwen3-embedding"
	}
	jsonDims, err := resolveDockerDimensions()
	if err != nil {
		return nil, nil, err
	}
	openEmb := NewOpenAIEmbedder(base, "", model, jsonDims)
	ep := strings.TrimRight(base, "/") + "/embeddings"
	SetActive("docker", model, Dimensions, "dmr", ep)
	return openEmb, func() bool { return true }, nil
}

func normalizeDMRBase(raw string) string {
	base := normalizeHTTPBase(raw, "http://127.0.0.1:12434")
	if strings.HasSuffix(base, "/engines/v1") {
		return base
	}
	if strings.HasSuffix(base, "/engines") {
		return base + "/v1"
	}
	if strings.HasSuffix(base, "/v1") {
		return strings.TrimSuffix(base, "/v1") + "/engines/v1"
	}
	return base + "/engines/v1"
}

func normalizeHTTPBase(raw, fallback string) string {
	url := strings.TrimSpace(raw)
	if url == "" {
		return fallback
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		url = "http://" + url
	}
	return strings.TrimRight(url, "/")
}

func resolveDockerDimensions() (int, error) {
	if _, ok := os.LookupEnv("EMBED_DOCKER_DIMENSIONS"); ok {
		v := strings.TrimSpace(os.Getenv("EMBED_DOCKER_DIMENSIONS"))
		return parseOpenAIDimensionsValue(v)
	}
	v := strings.TrimSpace(db.GetSetting("EMBED_DOCKER_DIMENSIONS", ""))
	if v == "" {
		return 768, nil
	}
	return parseOpenAIDimensionsValue(v)
}
