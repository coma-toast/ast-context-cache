package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/hostdisk"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

type batchResponse struct {
	Status   string            `json:"status"`
	Values   map[string]string `json:"values"`
	Reloaded bool              `json:"reloaded"`
	Error    string            `json:"error"`
	Field    string            `json:"field"`
	Errors   map[string]string `json:"errors"`
}

// setupSettingsBatch opens a fresh database with no embed env overrides, AST_INSTANCE_NAME set
// to instanceEnv, and counts embedder reloads instead of building one.
func setupSettingsBatch(t *testing.T, instanceEnv string) (http.Handler, *int) {
	t.Helper()
	for _, k := range embedder.SettingKeys() {
		if v, ok := os.LookupEnv(k); ok {
			t.Cleanup(func() { _ = os.Setenv(k, v) })
			os.Unsetenv(k)
		}
	}
	reloads, prev := 0, reloadEmbedder
	reloadEmbedder = func() error {
		reloads++
		return nil
	}
	t.Cleanup(func() { reloadEmbedder = prev })
	return setupInstanceAPI(t, instanceEnv), &reloads
}

func postBatch(t *testing.T, h http.Handler, method, body string) (int, batchResponse) {
	t.Helper()
	req := httptest.NewRequest(method, "/api/settings/batch", strings.NewReader(body))
	req.Host = "127.0.0.1:7830"
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var out batchResponse
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &out), rr.Body.String())
	return rr.Code, out
}

func TestSettingsBatch(t *testing.T) {
	tests := []struct {
		name        string
		instanceEnv string
		method      string
		body        string
		wantCode    int
		wantValues  map[string]string
		wantErrors  []string
		wantStored  map[string]string
		wantReloads int
	}{
		{
			name:       "all valid keys are normalized and stored",
			body:       `{"settings":{"idle_unload_minutes":" 5 ","context_max_notes_session":"10","handoff_max_depth":"3"}}`,
			wantCode:   http.StatusOK,
			wantValues: map[string]string{"idle_unload_minutes": "5", "context_max_notes_session": "10", "handoff_max_depth": "3"},
			wantStored: map[string]string{"idle_unload_minutes": "5", "context_max_notes_session": "10", "handoff_max_depth": "3"},
		},
		{
			name:       "one invalid key writes nothing",
			body:       `{"settings":{"idle_unload_minutes":"7","embed_worker_max":"0"}}`,
			wantCode:   http.StatusBadRequest,
			wantErrors: []string{"embed_worker_max"},
			wantStored: map[string]string{"idle_unload_minutes": "", "embed_worker_max": ""},
		},
		{
			name:       "every invalid key is reported",
			body:       `{"settings":{"idle_unload_minutes":"-1","watcher_ignore_globs":"not json","index_log_files":"yes","EMBED_BACKEND":"nope"}}`,
			wantCode:   http.StatusBadRequest,
			wantErrors: []string{"EMBED_BACKEND", "idle_unload_minutes", "index_log_files", "watcher_ignore_globs"},
			wantStored: map[string]string{"idle_unload_minutes": "", "watcher_ignore_globs": "", "index_log_files": "", "EMBED_BACKEND": ""},
		},
		{
			name:        "embed keys reload the embedder once",
			body:        `{"settings":{"EMBED_BACKEND":"http","EMBED_HTTP_URL":"http://batch.example/embed","EMBED_HTTP_BEARER":"s3cret"}}`,
			wantCode:    http.StatusOK,
			wantValues:  map[string]string{"EMBED_BACKEND": "http", "EMBED_HTTP_URL": "http://batch.example/embed", "EMBED_HTTP_BEARER": ""},
			wantStored:  map[string]string{"EMBED_BACKEND": "http", "EMBED_HTTP_URL": "http://batch.example/embed", "EMBED_HTTP_BEARER": "s3cret"},
			wantReloads: 1,
		},
		{
			name:       "aux workers checked against the max in the same batch",
			body:       `{"settings":{"embed_aux_worker_max":"4","EMBED_AUX_WORKERS":"6"}}`,
			wantCode:   http.StatusBadRequest,
			wantErrors: []string{"EMBED_AUX_WORKERS"},
			wantStored: map[string]string{"embed_aux_worker_max": ""},
		},
		{
			name:        "env-locked name is a conflict",
			instanceEnv: "fromenv",
			body:        `{"settings":{"instance_name":"dash"}}`,
			wantCode:    http.StatusConflict,
			wantErrors:  []string{instance.SettingKey},
			wantStored:  map[string]string{instance.SettingKey: ""},
		},
		{
			name:        "override in the same batch unlocks the name",
			instanceEnv: "fromenv",
			body:        `{"settings":{"instance_name":"dash","instance_name_override":"true"}}`,
			wantCode:    http.StatusOK,
			wantValues:  map[string]string{instance.SettingKey: "dash", instance.OverrideKey: "true"},
			wantStored:  map[string]string{instance.SettingKey: "dash", instance.OverrideKey: "true"},
		},
		{
			name:       "host disk health URL is trimmed and stored",
			body:       `{"settings":{"host_disk_health_url":" http://192.168.1.10:9633/metrics "}}`,
			wantCode:   http.StatusOK,
			wantValues: map[string]string{hostdisk.SettingKey: "http://192.168.1.10:9633/metrics"},
			wantStored: map[string]string{hostdisk.SettingKey: "http://192.168.1.10:9633/metrics"},
		},
		{
			name:       "host disk health URL with credentials is refused",
			body:       `{"settings":{"host_disk_health_url":"http://root:hunter2@pve:9633/metrics"}}`,
			wantCode:   http.StatusBadRequest,
			wantErrors: []string{hostdisk.SettingKey},
			wantStored: map[string]string{hostdisk.SettingKey: ""},
		},
		{
			name:       "host disk health URL must be http or https",
			body:       `{"settings":{"host_disk_health_url":"pve:9633/metrics"}}`,
			wantCode:   http.StatusBadRequest,
			wantErrors: []string{hostdisk.SettingKey},
			wantStored: map[string]string{hostdisk.SettingKey: ""},
		},
		{name: "empty batch", body: `{"settings":{}}`, wantCode: http.StatusBadRequest},
		{name: "bad JSON", body: `{"settings":`, wantCode: http.StatusBadRequest},
		{name: "GET not allowed", method: http.MethodGet, wantCode: http.StatusMethodNotAllowed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, reloads := setupSettingsBatch(t, tt.instanceEnv)
			method := tt.method
			if method == "" {
				method = http.MethodPost
			}
			code, out := postBatch(t, h, method, tt.body)
			require.Equal(t, tt.wantCode, code, out.Error)
			if tt.wantCode == http.StatusOK {
				assert.Equal(t, "ok", out.Status)
				assert.Equal(t, tt.wantValues, out.Values)
			} else {
				assert.NotEmpty(t, out.Error)
			}
			if tt.wantErrors != nil {
				assert.ElementsMatch(t, tt.wantErrors, keys(out.Errors), out.Errors)
				assert.Equal(t, tt.wantErrors[0], out.Field)
			}
			for k, want := range tt.wantStored {
				assert.Equal(t, want, db.GetSetting(k, ""), k)
			}
			assert.Equal(t, tt.wantReloads, *reloads)
			assert.Equal(t, tt.wantReloads > 0, out.Reloaded)
		})
	}
}

func keys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestSettingsSingleKeyRejectsWith4xx(t *testing.T) {
	tests := []struct {
		name, key, value string
		wantCode         int
		wantErr          string
	}{
		{name: "missing key", key: "", value: "x", wantCode: http.StatusBadRequest, wantErr: "key required"},
		{name: "worker max out of range", key: "embed_worker_max", value: "0", wantCode: http.StatusBadRequest, wantErr: "embed_worker_max must be 1–"},
		{name: "probe interval out of range", key: embedder.ProbeIntervalSettingKey, value: "1", wantCode: http.StatusBadRequest, wantErr: "must be 5–600"},
		{name: "host disk URL without scheme", key: hostdisk.SettingKey, value: "pve:9633", wantCode: http.StatusBadRequest, wantErr: "http:// or https://"},
		{name: "host disk URL clears", key: hostdisk.SettingKey, value: "", wantCode: http.StatusOK},
		{name: "valid key saves", key: "query_retention_max_age_days", value: "30", wantCode: http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, _ := setupSettingsBatch(t, "")
			code, out := postSetting(t, h, tt.key, tt.value)
			require.Equal(t, tt.wantCode, code, out)
			if tt.wantErr != "" {
				assert.Contains(t, out["error"], tt.wantErr)
				assert.Equal(t, tt.key, out["field"])
				return
			}
			assert.Equal(t, tt.key, out["key"])
			assert.Equal(t, tt.value, out["value"])
		})
	}
}

func TestDashboardSettingsRedactsEmbedSecrets(t *testing.T) {
	h, _ := setupSettingsBatch(t, "")
	code, out := postBatch(t, h, http.MethodPost, `{"settings":{"EMBED_OPENAI_API_KEY":"sk-batch"}}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	req := httptest.NewRequest(http.MethodGet, "/api/dashboard/settings", nil)
	req.Host = "127.0.0.1:7830"
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)
	assert.NotContains(t, rr.Body.String(), "sk-batch")
	var data map[string]interface{}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &data))
	assert.Equal(t, true, data["EmbedOpenAIAPIKeySet"])
	assert.Equal(t, false, data["EmbedHTTPBearerSet"])
	assert.Equal(t, "", data["EmbedOpenAIAPIKey"])
}

func TestRedactSecret(t *testing.T) {
	tests := []struct{ name, key, value, msg, want string }{
		{"secret value removed", "EMBED_OPENAI_API_KEY", " sk-1 ", "bad key sk-1 here", "bad key [redacted] here"},
		{"plain key kept", "idle_unload_minutes", "x", "x is bad", "x is bad"},
		{"empty secret kept", "EMBED_HTTP_BEARER", "", "bearer required", "bearer required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, redactSecret(tt.key, tt.value, tt.msg))
		})
	}
}

func TestSettingsBatchAutoLogsConfigChangeAndMasksSecrets(t *testing.T) {
	h, _ := setupSettingsBatch(t, "")
	rawSecret := "sk-supersecret-never-leak-987654321"

	// 1. Initial save: non-secret and secret keys both record config_change; secret is masked.
	code, out := postBatch(t, h, http.MethodPost, `{"settings":{"idle_unload_minutes":"5","EMBED_OPENAI_API_KEY":"`+rawSecret+`"}}`)
	require.Equal(t, http.StatusOK, code, out.Error)

	res, err := memory.QueryKB(memory.KBQueryInput{
		Kinds:   []memory.Kind{memory.KindConfigChange},
		Scope:   memory.ScopeGlobal,
		Status:  "all",
		PerPage: 50,
	})
	require.NoError(t, err)
	require.Len(t, res.Entries, 2)

	byTarget := map[string]memory.Entry{}
	for _, e := range res.Entries {
		byTarget[e.Target] = e
		assert.Equal(t, "dashboard", e.Actor)
		assert.Equal(t, memory.ScopeGlobal, e.Scope)
		assert.Equal(t, defaultConfigChangeReason, e.Reason)
	}
	assert.Equal(t, "", byTarget["idle_unload_minutes"].BeforeValue)
	assert.Equal(t, "5", byTarget["idle_unload_minutes"].AfterValue)
	assert.Equal(t, "(unset)", byTarget["EMBED_OPENAI_API_KEY"].BeforeValue)
	assert.Equal(t, "(updated)", byTarget["EMBED_OPENAI_API_KEY"].AfterValue)

	var leakCount int
	require.NoError(t, db.ContextDB.QueryRow(
		`SELECT COUNT(*) FROM structured_memory WHERE before_value LIKE ? OR after_value LIKE ? OR body LIKE ? OR reason LIKE ?`,
		"%"+rawSecret+"%", "%"+rawSecret+"%", "%"+rawSecret+"%", "%"+rawSecret+"%",
	).Scan(&leakCount))
	assert.Zero(t, leakCount, "raw secret must never be persisted in structured_memory")

	// 2. Unchanged non-secret value (even with whitespace) must not log another entry.
	code, out = postBatch(t, h, http.MethodPost, `{"settings":{"idle_unload_minutes":" 5 "}}`)
	require.Equal(t, http.StatusOK, code, out.Error)

	req := httptest.NewRequest(http.MethodPost, "/api/settings/batch", nil)
	applied, _, err := applySettings(httptest.NewRecorder(), req, map[string]string{"idle_unload_minutes": "5"})
	require.NoError(t, err)
	assert.Zero(t, applied.notify&realtime.KnowledgeBase, "unchanged setting must not notify KnowledgeBase")

	res, err = memory.QueryKB(memory.KBQueryInput{
		Kinds:   []memory.Kind{memory.KindConfigChange},
		Scope:   memory.ScopeGlobal,
		Status:  "all",
		PerPage: 50,
	})
	require.NoError(t, err)
	require.Len(t, res.Entries, 2)

	// 3. Update non-secret and clear secret: records before -> after and (set) -> (cleared), and notifies KnowledgeBase.
	applied, _, err = applySettings(httptest.NewRecorder(), req, map[string]string{
		"idle_unload_minutes":  "10",
		"EMBED_OPENAI_API_KEY": "",
	})
	require.NoError(t, err)
	assert.NotZero(t, applied.notify&realtime.KnowledgeBase, "changed settings must notify KnowledgeBase")

	histIdle, err := memory.History(memory.HistoryInput{Ref: byTarget["idle_unload_minutes"].Ref})
	require.NoError(t, err)
	require.Len(t, histIdle.Entries, 2)
	assert.Equal(t, "5", histIdle.Entries[1].BeforeValue)
	assert.Equal(t, "10", histIdle.Entries[1].AfterValue)
	assert.Equal(t, histIdle.Entries[1].Ref, histIdle.Entries[0].SupersededBy)

	histKey, err := memory.History(memory.HistoryInput{Ref: byTarget["EMBED_OPENAI_API_KEY"].Ref})
	require.NoError(t, err)
	require.Len(t, histKey.Entries, 2)
	assert.Equal(t, "(set)", histKey.Entries[1].BeforeValue)
	assert.Equal(t, "(cleared)", histKey.Entries[1].AfterValue)
	assert.Equal(t, histKey.Entries[1].Ref, histKey.Entries[0].SupersededBy)

	// 4. Clearing an already unset secret must not log.
	applied, _, err = applySettings(httptest.NewRecorder(), req, map[string]string{"EMBED_OPENAI_API_KEY": ""})
	require.NoError(t, err)
	assert.Zero(t, applied.notify&realtime.KnowledgeBase)

	histKeyAfter, err := memory.History(memory.HistoryInput{Ref: byTarget["EMBED_OPENAI_API_KEY"].Ref})
	require.NoError(t, err)
	assert.Len(t, histKeyAfter.Entries, 2)
}

func TestBuildMemoryScopedCounts(t *testing.T) {
	_, _ = setupSettingsBatch(t, "")

	_, err := memory.Store(memory.StoreInput{
		Kind:        memory.KindFact,
		Scope:       memory.ScopeProject,
		ProjectPath: "/proj/a",
		Subject:     "api",
		Predicate:   "uses",
		Object:      "rest",
	})
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{
		Kind:        memory.KindProcedure,
		Scope:       memory.ScopeProject,
		ProjectPath: "/proj/a",
		Rule:        "always run go test",
	})
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{
		Kind:   memory.KindPreference,
		Scope:  memory.ScopeGlobal,
		Object: "concise output",
	})
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{
		Kind:        memory.KindEvent,
		Scope:       memory.ScopeProject,
		ProjectPath: "/proj/a",
		Title:       "deployed v1",
	})
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{
		Kind:       memory.KindConfigChange,
		Scope:      memory.ScopeGlobal,
		Target:     "idle_unload_minutes",
		AfterValue: "5",
	})
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{
		Kind:        memory.KindDiscovery,
		Scope:       memory.ScopeProject,
		ProjectPath: "/proj/a",
		Title:       "cache warm",
	})
	require.NoError(t, err)
	// Different project entry should be excluded when filtering to /proj/a.
	_, err = memory.Store(memory.StoreInput{
		Kind:        memory.KindFact,
		Scope:       memory.ScopeProject,
		ProjectPath: "/proj/b",
		Subject:     "other",
		Predicate:   "uses",
		Object:      "grpc",
	})
	require.NoError(t, err)
	// Quarantined entry in /proj/a should count in QuarantinedCount, not ActiveFacts.
	qRes, err := memory.Store(memory.StoreInput{
		Kind:        memory.KindFact,
		Scope:       memory.ScopeProject,
		ProjectPath: "/proj/a",
		Subject:     "q",
		Predicate:   "is",
		Object:      "pending",
	})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET quarantined = 1 WHERE ref = ?`, qRes.Ref)
	require.NoError(t, err)

	scoped := buildMemory("/proj/a", 1)
	assert.Equal(t, 1, scoped.ActiveFacts)
	assert.Equal(t, 1, scoped.ActiveProcedures)
	assert.Equal(t, 1, scoped.ActivePreferences)
	assert.Equal(t, 1, scoped.ActiveEvents)
	assert.Equal(t, 1, scoped.ActiveConfigChanges)
	assert.Equal(t, 1, scoped.ActiveDiscoveries)
	assert.Equal(t, 1, scoped.QuarantinedCount)
	assert.Greater(t, scoped.StructuredMemoryTokens, 0)
	assert.Equal(t, 6, scoped.MemoryOrphanCount)

	all := buildMemory("", 1)
	assert.Equal(t, 2, all.ActiveFacts)
	assert.Greater(t, all.StructuredMemoryTokens, scoped.StructuredMemoryTokens)
}
