package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/flags"
)

// No t.Parallel: the db pools and the flags snapshot are package globals.

type flagsResponse struct {
	Status string            `json:"status"`
	Error  string            `json:"error"`
	Flags  []flags.FlagState `json:"flags"`
}

// setupFlagsAPI clears every flag env var (one left set in the developer's shell would lock a
// flag), opens a fresh database, and rebuilds the flags snapshot from it. The Reload cleanup is
// registered before dbtest.Init so it runs after the database closes, dropping the snapshot.
func setupFlagsAPI(t *testing.T) http.Handler {
	t.Helper()
	for _, f := range flags.All() {
		t.Setenv(f.Env, "")
	}
	t.Cleanup(flags.Reload)
	dbtest.Init(t)
	flags.Reload()
	return NewHandler("127.0.0.1")
}

// serveFlags sends an Origin-less request, as a CLI or curl would.
func serveFlags(t *testing.T, h http.Handler, method, path, body string) (int, flagsResponse) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "127.0.0.1:7830"
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var out flagsResponse
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &out), rr.Body.String())
	return rr.Code, out
}

func flagByKey(t *testing.T, states []flags.FlagState, key string) flags.FlagState {
	t.Helper()
	for _, st := range states {
		if st.Key == key {
			return st
		}
	}
	t.Fatalf("no flag %s in %+v", key, states)
	return flags.FlagState{}
}

func TestFlagsAPIListsEveryFlag(t *testing.T) {
	h := setupFlagsAPI(t)
	code, out := serveFlags(t, h, http.MethodGet, "/api/dashboard/flags", "")
	require.Equal(t, http.StatusOK, code)
	all := flags.All()
	require.Len(t, out.Flags, len(all))
	require.Len(t, out.Flags, 6)
	for i, f := range all {
		st := out.Flags[i]
		assert.Equal(t, f.Key, st.Key, "registry order")
		assert.Equal(t, f.Env, st.Env, f.Key)
		assert.Equal(t, f.Default, st.Enabled, f.Key)
		assert.Equal(t, flags.SourceDefault, st.Source, f.Key)
		assert.False(t, st.Locked, f.Key)
		assert.NotEmpty(t, st.Description, f.Key)
	}
}

func TestFlagsAPIToggle(t *testing.T) {
	h := setupFlagsAPI(t)
	code, out := serveFlags(t, h, http.MethodPost, "/api/dashboard/flags", `{"key":"feature_handoff_hooks","enabled":true}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	assert.Equal(t, "ok", out.Status)
	hooks := flagByKey(t, out.Flags, flags.KeyHandoffHooks)
	assert.True(t, hooks.Enabled)
	assert.Equal(t, flags.SourceSetting, hooks.Source)
	_, got := serveFlags(t, h, http.MethodGet, "/api/dashboard/flags", "")
	assert.True(t, flagByKey(t, got.Flags, flags.KeyHandoffHooks).Enabled, "GET reflects the POST")
	assert.True(t, flags.Enabled(flags.KeyHandoffHooks))
	code, out = serveFlags(t, h, http.MethodPost, "/api/dashboard/flags", `{"key":"feature_handoff","enabled":false}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	child := flagByKey(t, out.Flags, flags.KeyHandoffScratchpad)
	assert.False(t, child.Enabled, "master switch off forces children off")
	assert.Equal(t, flags.SourceDefault, child.Source, "child keeps its own source")
}

// TestFlagsAPIEnvLocked covers AC29: an env-set flag reports source env, is locked, and
// refuses dashboard writes with 409.
func TestFlagsAPIEnvLocked(t *testing.T) {
	h := setupFlagsAPI(t)
	t.Setenv("AST_FEATURE_HANDOFF_HOOKS", "true")
	flags.Reload()
	code, out := serveFlags(t, h, http.MethodPost, "/api/dashboard/flags", `{"key":"feature_handoff_hooks","enabled":false}`)
	require.Equal(t, http.StatusConflict, code)
	assert.Contains(t, out.Error, "locked by environment")
	_, got := serveFlags(t, h, http.MethodGet, "/api/dashboard/flags", "")
	hooks := flagByKey(t, got.Flags, flags.KeyHandoffHooks)
	assert.Equal(t, flags.SourceEnv, hooks.Source)
	assert.True(t, hooks.Locked)
	assert.True(t, hooks.Enabled)
	assert.Empty(t, db.GetSetting(flags.KeyHandoffHooks, ""), "a refused write is not persisted")
}

func TestFlagsAPIRejectsBadRequests(t *testing.T) {
	h := setupFlagsAPI(t)
	tests := []struct {
		name   string
		method string
		body   string
		want   int
	}{
		{"unknown key", http.MethodPost, `{"key":"feature_nope","enabled":true}`, http.StatusNotFound},
		{"invalid json", http.MethodPost, `{`, http.StatusBadRequest},
		{"missing enabled", http.MethodPost, `{"key":"feature_handoff"}`, http.StatusBadRequest},
		{"missing key", http.MethodPost, `{"enabled":true}`, http.StatusBadRequest},
		{"wrong method", http.MethodPut, `{}`, http.StatusMethodNotAllowed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, out := serveFlags(t, h, tt.method, "/api/dashboard/flags", tt.body)
			assert.Equal(t, tt.want, code)
			assert.NotEmpty(t, out.Error)
		})
	}
}

func TestSettingsPostRejectsFlagKeys(t *testing.T) {
	h := setupFlagsAPI(t)
	code, out := serveFlags(t, h, http.MethodPost, "/api/settings", `{"key":"feature_handoff","value":"false"}`)
	require.Equal(t, http.StatusBadRequest, code)
	assert.Equal(t, "use /api/dashboard/flags for feature flags", out.Error)
	assert.Empty(t, db.GetSetting(flags.KeyHandoff, ""), "flag not written through the generic path")
	assert.True(t, flags.Enabled(flags.KeyHandoff))
}
