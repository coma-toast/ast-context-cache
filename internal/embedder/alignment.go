package embedder

import (
	"os"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Alignment compares the running embedder (process start) with effective settings (env + DB).
type Alignment struct {
	InSync            bool     `json:"in_sync"`
	ActiveBackend     string   `json:"active_backend"`
	ActiveModel       string   `json:"active_model"`
	ConfiguredBackend string   `json:"configured_backend"`
	ConfiguredModel   string   `json:"configured_model"`
	EnvOverrides      []string `json:"env_overrides,omitempty"`
}

// EffectiveSettings returns embed config as NewForMain would read at runtime.
func EffectiveSettings() Settings {
	return Settings{
		Backend:          EffectiveEnv("EMBED_BACKEND"),
		ModelDir:         EffectiveEnv("MODEL_DIR"),
		HTTPURL:          EffectiveEnv("EMBED_HTTP_URL"),
		HTTPBearer:       EffectiveEnv("EMBED_HTTP_BEARER"),
		OllamaHost:       EffectiveEnv("OLLAMA_HOST"),
		OllamaModel:      EffectiveEnv("OLLAMA_EMBED_MODEL"),
		OpenAIBaseURL:    EffectiveEnv("EMBED_OPENAI_BASE_URL"),
		OpenAIAPIKey:     EffectiveEnv("EMBED_OPENAI_API_KEY"),
		OpenAIModel:      EffectiveEnv("EMBED_OPENAI_MODEL"),
		OpenAIDimensions: effectiveOpenAIDimensionsSetting(),
		DockerURL:        EffectiveEnv("EMBED_DOCKER_URL"),
		DockerModel:      EffectiveEnv("EMBED_DOCKER_MODEL"),
		DockerDimensions: EffectiveEnv("EMBED_DOCKER_DIMENSIONS"),
	}
}

func effectiveOpenAIDimensionsSetting() string {
	if _, ok := os.LookupEnv("EMBED_OPENAI_DIMENSIONS"); ok {
		return strings.TrimSpace(os.Getenv("EMBED_OPENAI_DIMENSIONS"))
	}
	return strings.TrimSpace(db.GetSetting("EMBED_OPENAI_DIMENSIONS", ""))
}

// ConfiguredSnapshot returns backend/model metadata for effective settings (post-restart target).
func ConfiguredSnapshot() (backend, model, runtime, endpoint string, dim int) {
	return SnapshotForSettings(EffectiveSettings())
}

// SnapshotForSettings derives display metadata without constructing an embedder.
func SnapshotForSettings(s Settings) (backend, model, runtime, endpoint string, dim int) {
	backend = normalizeEmbedBackend(s.Backend)
	dim = Dimensions
	switch backend {
	case "onnx":
		return "onnx", ModelName, "onnxruntime", "", dim
	case "http":
		u := strings.TrimSpace(s.HTTPURL)
		if u == "" {
			u = "http://127.0.0.1:8080/embed"
		}
		return "http", "remote", "http", u, dim
	case "ollama":
		url := strings.TrimSpace(s.OllamaHost)
		if url != "" && !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			url = "http://" + url
		}
		if url == "" {
			url = "http://127.0.0.1:11434"
		}
		model = strings.TrimSpace(s.OllamaModel)
		if model == "" {
			model = "nomic-embed-text"
		}
		return "ollama", model, "ollama", url + "/api/embed", dim
	case "openai":
		base := strings.TrimSpace(s.OpenAIBaseURL)
		if base == "" {
			base = "https://api.openai.com/v1"
		}
		model = strings.TrimSpace(s.OpenAIModel)
		if model == "" {
			model = "(unset)"
		}
		ep := strings.TrimRight(base, "/") + "/embeddings"
		return "openai", model, "openai", ep, dim
	case "docker":
		url := strings.TrimSpace(s.DockerURL)
		if url == "" {
			url = DefaultDockerURL
		}
		model = strings.TrimSpace(s.DockerModel)
		if model == "" {
			model = DefaultDockerModel
		}
		base := normalizeDMRBase(url)
		ep := strings.TrimRight(base, "/") + "/embeddings"
		return "docker", model, "dmr", ep, dim
	default:
		return backend, "", "", "", dim
	}
}

// AlignmentStatus compares ActiveSnapshot with effective configured settings.
func AlignmentStatus() Alignment {
	ab, am, _, _, _ := ActiveSnapshot()
	cb, cm, _, _, _ := ConfiguredSnapshot()
	return Alignment{
		InSync:            embedSnapshotsMatch(ab, am, cb, cm),
		ActiveBackend:     ab,
		ActiveModel:       am,
		ConfiguredBackend: cb,
		ConfiguredModel:   cm,
		EnvOverrides:      EnvOverrides(),
	}
}

func normalizeEmbedBackend(b string) string {
	switch strings.ToLower(strings.TrimSpace(b)) {
	case "", "onnx":
		return "onnx"
	case "litellm":
		return "openai"
	default:
		return strings.ToLower(strings.TrimSpace(b))
	}
}

func embedSnapshotsMatch(activeBackend, activeModel, configuredBackend, configuredModel string) bool {
	if normalizeEmbedBackend(activeBackend) != normalizeEmbedBackend(configuredBackend) {
		return false
	}
	am := strings.TrimSpace(activeModel)
	cm := strings.TrimSpace(configuredModel)
	if cm == "(unset)" {
		return false
	}
	return am == cm
}

// VerifyRunning probes the live process embedder and returns alignment + probe result.
type VerifyRunningResult struct {
	OK           bool   `json:"ok"`
	InSync       bool   `json:"in_sync"`
	Backend      string `json:"backend"`
	Model        string `json:"model"`
	Endpoint     string `json:"endpoint,omitempty"`
	Dimensions   int    `json:"dimensions"`
	LatencyMs    int64  `json:"latency_ms"`
	Error        string `json:"error,omitempty"`
	EnvOverrides []string `json:"env_overrides,omitempty"`
	ConfiguredBackend string `json:"configured_backend"`
	ConfiguredModel   string `json:"configured_model"`
}

func VerifyRunning(embed Interface) VerifyRunningResult {
	align := AlignmentStatus()
	res := VerifyRunningResult{
		InSync:            align.InSync,
		Backend:           align.ActiveBackend,
		Model:             align.ActiveModel,
		EnvOverrides:      align.EnvOverrides,
		ConfiguredBackend: align.ConfiguredBackend,
		ConfiguredModel:   align.ConfiguredModel,
	}
	_, _, _, ep, _ := ActiveSnapshot()
	res.Endpoint = ep
	if embed == nil {
		res.Error = "embedder not available"
		return res
	}
	start := time.Now()
	vecs, err := embed.Embed([]string{"embedding verify"})
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
	res.Dimensions = len(vecs[0])
	return res
}
