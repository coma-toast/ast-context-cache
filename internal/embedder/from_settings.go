package embedder

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultDockerURL is the DMR host when EMBED_DOCKER_URL is unset.
const (
	DefaultDockerURL         = "http://127.0.0.1:12434"
	DefaultDockerModel       = "ai/qwen3-embedding"
	DefaultDockerDimensions  = "768"
)

// Settings holds embedding config from dashboard form values (not process env).
type Settings struct {
	Backend            string
	ModelDir           string
	HTTPURL            string
	HTTPBearer         string
	OllamaHost         string
	OllamaModel        string
	OpenAIBaseURL      string
	OpenAIAPIKey       string
	OpenAIModel        string
	OpenAIDimensions   string
	DockerURL          string
	DockerModel        string
	DockerDimensions   string
}

// SettingsFromMap parses dashboard/API settings keys into a Settings value.
func SettingsFromMap(m map[string]string) Settings {
	return Settings{
		Backend:          strings.TrimSpace(m["EMBED_BACKEND"]),
		ModelDir:         strings.TrimSpace(m["MODEL_DIR"]),
		HTTPURL:          strings.TrimSpace(m["EMBED_HTTP_URL"]),
		HTTPBearer:       strings.TrimSpace(m["EMBED_HTTP_BEARER"]),
		OllamaHost:       strings.TrimSpace(m["OLLAMA_HOST"]),
		OllamaModel:      strings.TrimSpace(m["OLLAMA_EMBED_MODEL"]),
		OpenAIBaseURL:    strings.TrimSpace(m["EMBED_OPENAI_BASE_URL"]),
		OpenAIAPIKey:     strings.TrimSpace(m["EMBED_OPENAI_API_KEY"]),
		OpenAIModel:      strings.TrimSpace(m["EMBED_OPENAI_MODEL"]),
		OpenAIDimensions: strings.TrimSpace(m["EMBED_OPENAI_DIMENSIONS"]),
		DockerURL:        strings.TrimSpace(m["EMBED_DOCKER_URL"]),
		DockerModel:      strings.TrimSpace(m["EMBED_DOCKER_MODEL"]),
		DockerDimensions: strings.TrimSpace(m["EMBED_DOCKER_DIMENSIONS"]),
	}
}

// NewFromSettings builds an embedder from explicit settings (dashboard test / preview).
func NewFromSettings(s Settings, modelDir string) (Interface, error) {
	backend := strings.ToLower(strings.TrimSpace(s.Backend))
	if backend == "" {
		backend = "onnx"
	}
	if modelDir == "" {
		modelDir = strings.TrimSpace(s.ModelDir)
	}
	if modelDir == "" {
		exePath, _ := os.Executable()
		modelDir = filepath.Join(filepath.Dir(exePath), "model")
	}
	switch backend {
	case "onnx":
		SetActive("onnx", ModelName, Dimensions, "onnxruntime", "")
		if err := EnsureModel(modelDir); err != nil {
			return nil, fmt.Errorf("model files: %w", err)
		}
		return New(modelDir)
	case "http":
		u := strings.TrimSpace(s.HTTPURL)
		if u == "" {
			u = "http://127.0.0.1:8080/embed"
		}
		SetActive("http", "remote", Dimensions, "http", u)
		return NewHTTPEmbedder(u, s.HTTPBearer), nil
	case "ollama":
		url := strings.TrimSpace(s.OllamaHost)
		if url != "" && !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			url = "http://" + url
		}
		if url == "" {
			url = "http://127.0.0.1:11434"
		}
		model := strings.TrimSpace(s.OllamaModel)
		if model == "" {
			model = "nomic-embed-text"
		}
		SetActive("ollama", model, Dimensions, "ollama", url+"/api/embed")
		return NewOllamaEmbedder(url, model), nil
	case "openai", "litellm":
		base := strings.TrimSpace(s.OpenAIBaseURL)
		if base == "" {
			base = "https://api.openai.com/v1"
		}
		model := strings.TrimSpace(s.OpenAIModel)
		if model == "" {
			return nil, fmt.Errorf("EMBED_OPENAI_MODEL is required when EMBED_BACKEND=openai")
		}
		jsonDims, err := parseOpenAIDimensionsValue(strings.TrimSpace(s.OpenAIDimensions))
		if err != nil {
			return nil, err
		}
		if s.OpenAIDimensions == "" {
			jsonDims = 768
		}
		openEmb := NewOpenAIEmbedder(base, s.OpenAIAPIKey, model, jsonDims)
		ep := strings.TrimRight(strings.TrimSpace(base), "/") + "/embeddings"
		SetActive("openai", model, Dimensions, "openai", ep)
		return openEmb, nil
	case "docker":
		return newDockerEmbedderFromSettings(s)
	default:
		return nil, fmt.Errorf("EMBED_BACKEND: unknown %q (use onnx, http, ollama, openai, or docker)", backend)
	}
}

func newDockerEmbedderFromSettings(s Settings) (Interface, error) {
	url := strings.TrimSpace(s.DockerURL)
	if url == "" {
		url = DefaultDockerURL
	}
	model := strings.TrimSpace(s.DockerModel)
	if model == "" {
		model = DefaultDockerModel
	}
	jsonDims, err := dockerDimensionsFromSetting(s.DockerDimensions)
	if err != nil {
		return nil, err
	}
	base := normalizeDMRBase(url)
	openEmb := NewOpenAIEmbedderWithLabel(base, "", model, jsonDims, "dmr embed")
	ep := strings.TrimRight(base, "/") + "/embeddings"
	SetActive("docker", model, Dimensions, "dmr", ep)
	return openEmb, nil
}

func dockerDimensionsFromSetting(v string) (int, error) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 768, nil
	}
	return parseOpenAIDimensionsValue(v)
}

// TestResult is returned after a dashboard embedder connectivity test.
type TestResult struct {
	OK         bool     `json:"ok"`
	Backend    string   `json:"backend"`
	Model      string   `json:"model"`
	Endpoint   string   `json:"endpoint"`
	Dimensions int      `json:"dimensions"`
	LatencyMs  int64    `json:"latency_ms"`
	EnvBlocked []string `json:"env_overrides,omitempty"`
	Error      string   `json:"error,omitempty"`
}

var embedSettingKeys = []string{
	"EMBED_BACKEND", "MODEL_DIR",
	"EMBED_HTTP_URL", "EMBED_HTTP_BEARER",
	"OLLAMA_HOST", "OLLAMA_EMBED_MODEL",
	"EMBED_OPENAI_BASE_URL", "EMBED_OPENAI_API_KEY", "EMBED_OPENAI_MODEL", "EMBED_OPENAI_DIMENSIONS",
	"EMBED_DOCKER_URL", "EMBED_DOCKER_MODEL", "EMBED_DOCKER_DIMENSIONS",
}

// SettingKeys lists dashboard/env keys for embed configuration.
func SettingKeys() []string {
	return append([]string(nil), embedSettingKeys...)
}

// IsSettingKey reports whether key is an embed settings field.
func IsSettingKey(key string) bool {
	for _, k := range embedSettingKeys {
		if k == key {
			return true
		}
	}
	return false
}

// EnvOverrides lists embed settings keys overridden by non-empty process env.
func EnvOverrides() []string {
	var out []string
	for _, k := range embedSettingKeys {
		if strings.TrimSpace(os.Getenv(k)) != "" {
			out = append(out, k)
		}
	}
	return out
}

// TestSettings embeds a probe string using the given settings.
func TestSettings(s Settings, modelDir string) TestResult {
	res := TestResult{EnvBlocked: EnvOverrides()}
	e, err := NewFromSettings(s, modelDir)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	start := time.Now()
	vecs, err := e.Embed([]string{"embedding test"})
	res.LatencyMs = time.Since(start).Milliseconds()
	if err != nil {
		res.Error = err.Error()
		return res
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		res.Error = "empty embedding vector"
		return res
	}
	res.OK = true
	res.Backend = ActiveBackend
	res.Model = ActiveModel
	res.Endpoint = ActiveEndpoint
	res.Dimensions = len(vecs[0])
	return res
}
