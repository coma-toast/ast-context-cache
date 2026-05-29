package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

func testEmbedDB(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
}

func saveEmbedSetting(t *testing.T, key, value string) {
	t.Helper()
	if err := db.SetSetting(key, value); err != nil {
		t.Fatalf("SetSetting %s: %v", key, err)
	}
	onEmbedSettingChanged(key)
}

func loadEmbedSettingsMap() map[string]string {
	out := make(map[string]string, len(embedder.SettingKeys()))
	for _, k := range embedder.SettingKeys() {
		out[k] = db.GetSetting(k, "")
	}
	return out
}

func settingsDataField(data *components.SettingsData, key string) string {
	switch key {
	case "EMBED_BACKEND":
		return data.EmbedBackend
	case "MODEL_DIR":
		return data.EmbedModelDir
	case "EMBED_HTTP_URL":
		return data.EmbedHTTPURL
	case "EMBED_HTTP_BEARER":
		return data.EmbedHTTPBearer
	case "OLLAMA_HOST":
		return data.EmbedOllamaHost
	case "OLLAMA_EMBED_MODEL":
		return data.EmbedOllamaModel
	case "EMBED_OPENAI_BASE_URL":
		return data.EmbedOpenAIBaseURL
	case "EMBED_OPENAI_API_KEY":
		return data.EmbedOpenAIAPIKey
	case "EMBED_OPENAI_MODEL":
		return data.EmbedOpenAIModel
	case "EMBED_OPENAI_DIMENSIONS":
		return data.EmbedOpenAIDimensions
	case "EMBED_DOCKER_URL":
		return data.EmbedDockerURL
	case "EMBED_DOCKER_MODEL":
		return data.EmbedDockerModel
	case "EMBED_DOCKER_DIMENSIONS":
		return data.EmbedDockerDimensions
	default:
		return ""
	}
}

func TestEmbedSetting_saveAndLoad_eachKey(t *testing.T) {
	testEmbedDB(t)
	saveEmbedSetting(t, "EMBED_BACKEND", "onnx")
	want := map[string]string{
		"MODEL_DIR":               "/data/models",
		"EMBED_HTTP_URL":          "http://127.0.0.1:9090/embed",
		"EMBED_HTTP_BEARER":       "secret-bearer",
		"OLLAMA_HOST":             "http://127.0.0.1:11434",
		"OLLAMA_EMBED_MODEL":      "nomic-embed-text",
		"EMBED_OPENAI_BASE_URL":   "https://litellm.example/v1",
		"EMBED_OPENAI_API_KEY":    "sk-test",
		"EMBED_OPENAI_MODEL":      "text-embedding-3-small",
		"EMBED_OPENAI_DIMENSIONS": "768",
		"EMBED_DOCKER_URL":        "http://127.0.0.1:12434",
		"EMBED_DOCKER_MODEL":      "ai/qwen3-embedding",
		"EMBED_DOCKER_DIMENSIONS": "768",
	}
	for key, value := range want {
		saveEmbedSetting(t, key, value)
		if got := db.GetSetting(key, ""); got != value {
			t.Fatalf("db load %s: got %q want %q", key, got, value)
		}
		var data components.SettingsData
		PopulateEmbedSettings(loadEmbedSettingsMap(), &data)
		if got := settingsDataField(&data, key); got != value {
			t.Fatalf("PopulateEmbedSettings %s: got %q want %q", key, got, value)
		}
	}
}

func TestEmbedSetting_profileSnapshot_eachBackendKey(t *testing.T) {
	testEmbedDB(t)
	cases := []struct {
		backend string
		fields  map[string]string
	}{
		{
			backend: "onnx",
			fields:  map[string]string{"MODEL_DIR": "/onnx/dir"},
		},
		{
			backend: "http",
			fields: map[string]string{
				"EMBED_HTTP_URL":    "http://embed.local/embed",
				"EMBED_HTTP_BEARER": "tok",
			},
		},
		{
			backend: "ollama",
			fields: map[string]string{
				"OLLAMA_HOST":        "http://ollama.local:11434",
				"OLLAMA_EMBED_MODEL": "mxbai-embed-large",
			},
		},
		{
			backend: "openai",
			fields: map[string]string{
				"EMBED_OPENAI_BASE_URL":   "https://api.openai.com/v1",
				"EMBED_OPENAI_API_KEY":    "key",
				"EMBED_OPENAI_MODEL":      "embed-model",
				"EMBED_OPENAI_DIMENSIONS": "0",
			},
		},
		{
			backend: "docker",
			fields: map[string]string{
				"EMBED_DOCKER_URL":        "http://dmr.local:12434",
				"EMBED_DOCKER_MODEL":      "ai/custom",
				"EMBED_DOCKER_DIMENSIONS": "768",
			},
		},
	}
	for _, tc := range cases {
		saveEmbedSetting(t, "EMBED_BACKEND", tc.backend)
		for k, v := range tc.fields {
			saveEmbedSetting(t, k, v)
		}
		snapshotEmbedProfileFromDB(tc.backend)
		profiles := loadEmbedProfiles()
		got := profiles[components.EmbedBackendUI(tc.backend)]
		for k, want := range tc.fields {
			if got[k] != want {
				t.Fatalf("%s profile[%s]: got %q want %q", tc.backend, k, got[k], want)
			}
		}
	}
}

func TestEmbedSetting_backendSwitch_restoresEachProfile(t *testing.T) {
	testEmbedDB(t)
	saveEmbedSetting(t, "EMBED_BACKEND", "ollama")
	saveEmbedSetting(t, "OLLAMA_HOST", "http://ollama.saved:11434")
	saveEmbedSetting(t, "OLLAMA_EMBED_MODEL", "saved-ollama-model")
	snapshotEmbedProfileFromDB("ollama")

	if err := switchEmbedBackend("ollama", "docker", "docker"); err != nil {
		t.Fatal(err)
	}
	saveEmbedSetting(t, "EMBED_DOCKER_URL", "http://dmr.saved:12434")
	saveEmbedSetting(t, "EMBED_DOCKER_MODEL", "saved-docker-model")
	snapshotEmbedProfileFromDB("docker")

	if err := switchEmbedBackend("docker", "ollama", "ollama"); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("OLLAMA_HOST", ""); got != "http://ollama.saved:11434" {
		t.Fatalf("OLLAMA_HOST: got %q", got)
	}
	if got := db.GetSetting("OLLAMA_EMBED_MODEL", ""); got != "saved-ollama-model" {
		t.Fatalf("OLLAMA_EMBED_MODEL: got %q", got)
	}

	if err := switchEmbedBackend("ollama", "docker", "docker"); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("EMBED_DOCKER_MODEL", ""); got != "saved-docker-model" {
		t.Fatalf("EMBED_DOCKER_MODEL: got %q", got)
	}
}

func TestPersistEmbedSettings_savesAndLoadsAllKeys_sameBackend(t *testing.T) {
	testEmbedDB(t)
	if err := db.SetSetting("EMBED_BACKEND", "openai"); err != nil {
		t.Fatal(err)
	}
	payload := map[string]string{
		"EMBED_BACKEND":           "openai",
		"MODEL_DIR":               "/models",
		"EMBED_HTTP_URL":          "http://http.example/embed",
		"EMBED_HTTP_BEARER":       "bearer",
		"OLLAMA_HOST":             "http://ollama.example",
		"OLLAMA_EMBED_MODEL":      "ollama-model",
		"EMBED_OPENAI_BASE_URL":   "https://openai.example/v1",
		"EMBED_OPENAI_API_KEY":    "sk-abc",
		"EMBED_OPENAI_MODEL":      "text-embedding-3-large",
		"EMBED_OPENAI_DIMENSIONS": "1536",
		"EMBED_DOCKER_URL":        "http://docker.example:12434",
		"EMBED_DOCKER_MODEL":      "ai/docker-model",
		"EMBED_DOCKER_DIMENSIONS": "768",
	}
	if err := PersistEmbedSettings(payload); err != nil {
		t.Fatal(err)
	}
	for key, want := range payload {
		if key == "EMBED_BACKEND" {
			want = normalizeEmbedBackendValue(want)
		}
		if got := db.GetSetting(key, ""); got != want {
			t.Fatalf("%s: got %q want %q", key, got, want)
		}
	}
	var data components.SettingsData
	PopulateEmbedSettings(loadEmbedSettingsMap(), &data)
	for key, want := range payload {
		if key == "EMBED_BACKEND" {
			want = normalizeEmbedBackendValue(want)
		}
		if got := settingsDataField(&data, key); got != want {
			t.Fatalf("PopulateEmbedSettings %s: got %q want %q", key, got, want)
		}
	}
}

func TestPersistEmbedSettings_backendSwitch_restoresProfile(t *testing.T) {
	testEmbedDB(t)
	saveEmbedSetting(t, "EMBED_BACKEND", "http")
	saveEmbedSetting(t, "EMBED_HTTP_URL", "http://saved-http/embed")
	snapshotEmbedProfileFromDB("http")

	payload := map[string]string{
		"EMBED_BACKEND":      "ollama",
		"OLLAMA_HOST":        "http://new-from-form:11434",
		"OLLAMA_EMBED_MODEL": "form-model",
	}
	if err := PersistEmbedSettings(payload); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("OLLAMA_EMBED_MODEL", ""); got != "form-model" {
		t.Fatalf("after switch: got %q", got)
	}

	payload = map[string]string{"EMBED_BACKEND": "http"}
	if err := PersistEmbedSettings(payload); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("EMBED_HTTP_URL", ""); got != "http://saved-http/embed" {
		t.Fatalf("restored http url: got %q", got)
	}
}

func TestHandleEmbedSettings_POST_saveAndLoad(t *testing.T) {
	testEmbedDB(t)
	body := `{
		"EMBED_BACKEND":"docker",
		"EMBED_DOCKER_URL":"http://post.example:12434",
		"EMBED_DOCKER_MODEL":"ai/post-model",
		"EMBED_DOCKER_DIMENSIONS":"768"
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/settings/embed", strings.NewReader(body))
	w := httptest.NewRecorder()
	handleEmbedSettings(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	if got := db.GetSetting("EMBED_BACKEND", ""); got != "docker" {
		t.Fatalf("EMBED_BACKEND: %q", got)
	}
	if got := db.GetSetting("EMBED_DOCKER_MODEL", ""); got != "ai/post-model" {
		t.Fatalf("EMBED_DOCKER_MODEL: %q", got)
	}
	var data components.SettingsData
	PopulateEmbedSettings(loadEmbedSettingsMap(), &data)
	if data.EmbedDockerModel != "ai/post-model" {
		t.Fatalf("PopulateEmbedSettings docker model: %q", data.EmbedDockerModel)
	}
	raw := db.GetSetting(embedProfilesKey, "")
	if raw == "" {
		t.Fatal("expected embed_backend_profiles to be saved")
	}
	var profiles map[string]map[string]string
	if err := json.Unmarshal([]byte(raw), &profiles); err != nil {
		t.Fatal(err)
	}
	if profiles["docker"]["EMBED_DOCKER_MODEL"] != "ai/post-model" {
		t.Fatalf("profile docker model: %v", profiles["docker"])
	}
}

func TestPopulateEmbedSettings_dockerDisplayDefaults(t *testing.T) {
	var data components.SettingsData
	PopulateEmbedSettings(map[string]string{
		"EMBED_BACKEND": "docker",
	}, &data)
	if data.EmbedDockerURL != embedder.DefaultDockerURL {
		t.Fatalf("docker url default: %q", data.EmbedDockerURL)
	}
	if data.EmbedDockerModel != embedder.DefaultDockerModel {
		t.Fatalf("docker model default: %q", data.EmbedDockerModel)
	}
}

func TestHandleSettings_singleKey_saveAndLoad(t *testing.T) {
	testEmbedDB(t)
	saveEmbedSetting(t, "EMBED_BACKEND", "http")
	cases := map[string]string{
		"EMBED_HTTP_URL":    "http://single-key.example/embed",
		"EMBED_HTTP_BEARER": "single-bearer",
	}
	for key, want := range cases {
		body := `{"key":"` + key + `","value":"` + want + `"}`
		req := httptest.NewRequest(http.MethodPost, "/api/settings", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		handleSettings(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("%s POST status %d: %s", key, w.Code, w.Body.String())
		}
		if got := db.GetSetting(key, ""); got != want {
			t.Fatalf("%s db: got %q want %q", key, got, want)
		}
	}
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	w := httptest.NewRecorder()
	handleSettings(w, req)
	var loaded map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &loaded); err != nil {
		t.Fatal(err)
	}
	for key, want := range cases {
		if got := loaded[key]; got != want {
			t.Fatalf("GET %s: got %q want %q", key, got, want)
		}
	}
}

func TestPersistEmbedSettings_backendSwitch_emptyFormDoesNotWipeProfile(t *testing.T) {
	testEmbedDB(t)
	saveEmbedSetting(t, "EMBED_BACKEND", "openai")
	saveEmbedSetting(t, "EMBED_OPENAI_MODEL", "saved-model")
	saveEmbedSetting(t, "EMBED_OPENAI_BASE_URL", "https://saved.example/v1")
	snapshotEmbedProfileFromDB("openai")
	saveEmbedSetting(t, "EMBED_BACKEND", "ollama")

	payload := map[string]string{
		"EMBED_BACKEND":           "openai",
		"EMBED_OPENAI_BASE_URL":   "",
		"EMBED_OPENAI_API_KEY":    "",
		"EMBED_OPENAI_MODEL":      "",
		"EMBED_OPENAI_DIMENSIONS": "",
	}
	if err := PersistEmbedSettings(payload); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("EMBED_OPENAI_MODEL", ""); got != "saved-model" {
		t.Fatalf("model wiped: got %q", got)
	}
	if got := db.GetSetting("EMBED_OPENAI_BASE_URL", ""); got != "https://saved.example/v1" {
		t.Fatalf("base url wiped: got %q", got)
	}
}

func TestHandleSettings_backendSwitch_normalizesLitellm(t *testing.T) {
	testEmbedDB(t)
	body := `{"key":"EMBED_BACKEND","value":"litellm"}`
	req := httptest.NewRequest(http.MethodPost, "/api/settings", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handleSettings(w, req)
	if got := db.GetSetting("EMBED_BACKEND", ""); got != "openai" {
		t.Fatalf("got %q want openai", got)
	}
}

func TestPersistEmbedSettings_normalizesLitellmBackend(t *testing.T) {
	testEmbedDB(t)
	if err := PersistEmbedSettings(map[string]string{"EMBED_BACKEND": "litellm"}); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("EMBED_BACKEND", ""); got != "openai" {
		t.Fatalf("got %q want openai", got)
	}
}

func TestSwitchEmbedBackend_storesValueAsGiven(t *testing.T) {
	testEmbedDB(t)
	if err := switchEmbedBackend("onnx", "openai", "litellm"); err != nil {
		t.Fatal(err)
	}
	if got := db.GetSetting("EMBED_BACKEND", ""); got != "litellm" {
		t.Fatalf("switchEmbedBackend stores raw value: got %q", got)
	}
	if components.EmbedBackendUI("litellm") != "openai" {
		t.Fatalf("EmbedBackendUI should map litellm to openai profile bucket")
	}
}
