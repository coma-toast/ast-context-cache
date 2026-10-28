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
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/instance"
)

// No t.Parallel: the db pools, the instance flag name and the current name are package globals.

// setupInstanceAPI sets AST_INSTANCE_NAME to env (empty unsets it), clears the
// -name flag and opens a fresh database. Saves rotate httpguard's cookie name,
// so it is restored afterwards.
func setupInstanceAPI(t *testing.T, env string) http.Handler {
	t.Helper()
	t.Setenv("AST_INSTANCE_NAME", env)
	instance.Init("")
	dbtest.Init(t)
	instance.Update("", false, db.GetDataDir())
	prevCookie := httpguard.SessionCookieName
	t.Cleanup(func() {
		instance.Init("")
		instance.Update("", false, "")
		httpguard.SessionCookieName = prevCookie
		httpguard.SetInstanceName("")
	})
	return NewHandler("127.0.0.1")
}

func postInstanceSetting(t *testing.T, h http.Handler, key, value string) (int, map[string]string) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/settings", strings.NewReader(`{"key":"`+key+`","value":"`+value+`"}`))
	req.Host = "127.0.0.1:7830"
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var out map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &out), rr.Body.String())
	return rr.Code, out
}

func TestInstanceNameSave(t *testing.T) {
	cases := []struct {
		name, env    string
		override     string // posted first when set
		wantCode     int
		wantEffect   string
		wantSource   string
		wantOverride bool
	}{
		{name: "no env saves to settings", wantCode: http.StatusOK, wantEffect: "dash", wantSource: instance.SourceSettings},
		{name: "env without override is 409", env: "fromenv", wantCode: http.StatusConflict, wantEffect: "fromenv", wantSource: instance.SourceEnv},
		{name: "env with override off is 409", env: "fromenv", override: "false", wantCode: http.StatusConflict, wantEffect: "fromenv", wantSource: instance.SourceEnv},
		{name: "env with override saves", env: "fromenv", override: "true", wantCode: http.StatusOK, wantEffect: "dash", wantSource: instance.SourceOverride, wantOverride: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := setupInstanceAPI(t, tc.env)
			if tc.override != "" {
				code, out := postInstanceSetting(t, h, instance.OverrideKey, tc.override)
				require.Equal(t, http.StatusOK, code, out)
			}
			code, out := postInstanceSetting(t, h, instance.SettingKey, "dash")
			assert.Equal(t, tc.wantCode, code, out)
			if code == http.StatusOK {
				assert.Equal(t, tc.wantSource, out["instance_name_source"])
			}
			data := buildSettingsData(settingsBuildOpts{})
			assert.Equal(t, tc.wantEffect, data.InstanceNameEffective)
			assert.Equal(t, tc.wantEffect, instance.Current().Value)
			assert.Equal(t, tc.wantSource, data.InstanceNameSource)
			assert.Equal(t, tc.wantOverride, data.InstanceNameOverride)
			assert.Equal(t, tc.env, data.InstanceNameExternal)
			assert.Equal(t, tc.env != "" && !tc.wantOverride, data.InstanceNameLocked)
		})
	}
}

func TestInstanceNameOverrideToggle(t *testing.T) {
	h := setupInstanceAPI(t, "fromenv")
	code, out := postInstanceSetting(t, h, instance.OverrideKey, "maybe")
	assert.Equal(t, http.StatusBadRequest, code, out)

	// Turning the override on with no stored name keeps the env value.
	code, out = postInstanceSetting(t, h, instance.OverrideKey, "true")
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, "fromenv", out["instance_name_effective"])
	assert.Equal(t, instance.SourceEnv, out["instance_name_source"])
	code, out = postInstanceSetting(t, h, instance.SettingKey, "dash")
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, "dash", out["instance_name_effective"])

	// Unchecking reverts to the env value and keeps the stored name.
	code, out = postInstanceSetting(t, h, instance.OverrideKey, "false")
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, "fromenv", out["instance_name_effective"])
	assert.Equal(t, "dash", db.GetSetting(instance.SettingKey, ""))
	code, _ = postInstanceSetting(t, h, instance.SettingKey, "other")
	assert.Equal(t, http.StatusConflict, code)

	// The generic GET reports the source, the override and the env value.
	settings := map[string]string{}
	applyInstanceSettings(settings)
	assert.Equal(t, map[string]string{
		"instance_name":           "dash",
		"instance_name_override":  "false",
		"instance_name_effective": "fromenv",
		"instance_scope":          "fromenv",
		"instance_name_source":    instance.SourceEnv,
		"instance_name_external":  "fromenv",
		"instance_name_locked":    "true",
	}, settings)

	// Re-checking brings the stored name back, and it survives a restart.
	code, out = postInstanceSetting(t, h, instance.OverrideKey, "true")
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, "dash", out["instance_name_effective"])
	instance.Init("")
	instance.Update("", false, db.GetDataDir())
	name, override := StoredInstanceName()
	n := instance.Update(name, override, db.GetDataDir())
	assert.Equal(t, "dash", n.Value)
	assert.Equal(t, instance.SourceOverride, n.Source)
}
