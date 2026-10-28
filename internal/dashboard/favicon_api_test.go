package dashboard

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/instance"
)

// No t.Parallel: the db pools and the env vars are process-wide.

// setupFaviconAPI sets the favicon env vars (empty unsets them) and opens a fresh database.
func setupFaviconAPI(t *testing.T, envColor, envBadge string) http.Handler {
	t.Helper()
	t.Setenv(instance.FaviconColorEnv, envColor)
	t.Setenv(instance.FaviconBadgeEnv, envBadge)
	return setupInstanceAPI(t, "")
}

func postSetting(t *testing.T, h http.Handler, key, value string) (int, map[string]string) {
	t.Helper()
	body, err := json.Marshal(map[string]string{"key": key, "value": value})
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodPost, "/api/settings", bytes.NewReader(body))
	req.Host = "127.0.0.1:7830"
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var out map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &out), rr.Body.String())
	return rr.Code, out
}

func getFavicon(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.Host = "127.0.0.1:7830"
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	return rr
}

func pngDataURL(t *testing.T, pad int) (string, []byte) {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 16, 16))))
	b := append(buf.Bytes(), make([]byte, pad)...)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(b), b
}

func svgDataURL(svg string) string {
	return "data:image/svg+xml;base64," + base64.StdEncoding.EncodeToString([]byte(svg))
}

func TestFaviconModesRoundTrip(t *testing.T) {
	pngURL, pngBytes := pngDataURL(t, 0)
	cases := []struct {
		name     string
		posts    [][2]string
		wantMode string
		wantType string
		wantBody func(t *testing.T, body []byte)
	}{
		{
			name: "default", posts: [][2]string{{instance.FaviconModeKey, instance.FaviconDefault}}, wantMode: instance.FaviconDefault, wantType: instance.FaviconSVG,
			wantBody: func(t *testing.T, body []byte) { assert.Equal(t, defaultFaviconSVG(), body) },
		},
		{
			name:  "color with badge",
			posts: [][2]string{{instance.FaviconColorKey, "#0A0"}, {instance.FaviconBadgeKey, "M"}, {instance.FaviconModeKey, instance.FaviconColor}}, wantMode: instance.FaviconColor, wantType: instance.FaviconSVG,
			wantBody: func(t *testing.T, body []byte) {
				assert.Contains(t, string(body), `fill="#00aa00"`)
				assert.Contains(t, string(body), `>M</text>`)
			},
		},
		{
			name: "upload switches to upload mode", posts: [][2]string{{instance.FaviconUploadKey, pngURL}}, wantMode: instance.FaviconUpload, wantType: instance.FaviconPNG,
			wantBody: func(t *testing.T, body []byte) { assert.Equal(t, pngBytes, body) },
		},
		{
			name: "svg upload", posts: [][2]string{{instance.FaviconUploadKey, svgDataURL(`<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>`)}}, wantMode: instance.FaviconUpload, wantType: instance.FaviconSVG,
			wantBody: func(t *testing.T, body []byte) { assert.Contains(t, string(body), "<rect") },
		},
		{
			name: "clearing the upload goes back to default", posts: [][2]string{{instance.FaviconUploadKey, pngURL}, {instance.FaviconUploadKey, ""}}, wantMode: instance.FaviconDefault, wantType: instance.FaviconSVG,
			wantBody: func(t *testing.T, body []byte) { assert.Equal(t, defaultFaviconSVG(), body) },
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := setupFaviconAPI(t, "", "")
			var out map[string]string
			for _, p := range tc.posts {
				var code int
				code, out = postSetting(t, h, p[0], p[1])
				require.Equal(t, http.StatusOK, code, out)
			}
			assert.Equal(t, tc.wantMode, out["favicon_mode"])
			data := buildSettingsData(settingsBuildOpts{})
			assert.Equal(t, tc.wantMode, data.InstanceFaviconEffectiveMode)
			assert.Equal(t, out["favicon_url"], data.InstanceFaviconURL)
			rr := getFavicon(t, h, data.InstanceFaviconURL)
			require.Equal(t, http.StatusOK, rr.Code)
			assert.Equal(t, tc.wantType, rr.Header().Get("Content-Type"))
			assert.Equal(t, "nosniff", rr.Header().Get("X-Content-Type-Options"))
			assert.Contains(t, rr.Header().Get("Content-Security-Policy"), "sandbox")
			assert.Contains(t, rr.Header().Get("Cache-Control"), "immutable")
			tc.wantBody(t, rr.Body.Bytes())
			if tc.wantMode == instance.FaviconUpload {
				assert.Equal(t, tc.wantType, data.InstanceFaviconUploadType)
			}
		})
	}
}

func TestFaviconRejectsBadInput(t *testing.T) {
	tooBig, _ := pngDataURL(t, instance.FaviconMaxBytes)
	cases := []struct{ name, key, value, err string }{
		{name: "upload over 64 KB", key: instance.FaviconUploadKey, value: tooBig, err: "limit is 64 KB"},
		{name: "jpeg upload", key: instance.FaviconUploadKey, value: "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString([]byte("\xff\xd8\xff\xe0jpeg")), err: "must be a PNG, SVG or ICO"},
		{name: "script svg", key: instance.FaviconUploadKey, value: svgDataURL(`<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>`), err: "<script>"},
		{name: "onload svg", key: instance.FaviconUploadKey, value: svgDataURL(`<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>`), err: "event handler"},
		{name: "external svg", key: instance.FaviconUploadKey, value: svgDataURL(`<svg xmlns="http://www.w3.org/2000/svg"><use href="https://evil.example/a.svg#x"/></svg>`), err: "#fragments"},
		{name: "plain url", key: instance.FaviconUploadKey, value: "https://example.com/favicon.png", err: "data URL"},
		{name: "bad color", key: instance.FaviconColorKey, value: "red", err: "hex color"},
		{name: "long badge", key: instance.FaviconBadgeKey, value: "ABC", err: "1–2 characters"},
		{name: "bad mode", key: instance.FaviconModeKey, value: "rainbow", err: "default, color or upload"},
		{name: "upload mode without upload", key: instance.FaviconModeKey, value: instance.FaviconUpload, err: "upload a favicon"},
		{name: "bad override", key: instance.FaviconOverrideKey, value: "yes", err: "true or false"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := setupFaviconAPI(t, "", "")
			code, out := postSetting(t, h, tc.key, tc.value)
			assert.Equal(t, http.StatusBadRequest, code, out)
			assert.Contains(t, out["error"], tc.err)
			assert.Equal(t, "", db.GetSetting(tc.key, ""), "nothing stored")
		})
	}
}

func TestSettingsBodyLimit(t *testing.T) {
	h := setupFaviconAPI(t, "", "")
	code, out := postSetting(t, h, instance.FaviconUploadKey, "data:image/png;base64,"+strings.Repeat("A", settingsMaxBodyBytes))
	assert.Equal(t, http.StatusBadRequest, code, out)
	assert.Contains(t, out["error"], "too large")
}

func TestFaviconEnvLockAndOverride(t *testing.T) {
	cases := []struct {
		name, envColor, envBadge, override string
		wantCode                           int
		wantSource                         string
		wantLocked, wantOverride           bool
	}{
		{name: "no env saves", wantCode: http.StatusOK, wantSource: instance.SourceSettings},
		{name: "env color is 409", envColor: "#ff0000", wantCode: http.StatusConflict, wantSource: instance.SourceEnv, wantLocked: true},
		{name: "env badge is 409", envBadge: "E", wantCode: http.StatusConflict, wantSource: instance.SourceEnv, wantLocked: true},
		{name: "override off is 409", envColor: "#ff0000", override: "false", wantCode: http.StatusConflict, wantSource: instance.SourceEnv, wantLocked: true},
		{name: "override saves", envColor: "#ff0000", override: "true", wantCode: http.StatusOK, wantSource: instance.SourceOverride, wantOverride: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := setupFaviconAPI(t, tc.envColor, tc.envBadge)
			if tc.override != "" {
				code, out := postSetting(t, h, instance.FaviconOverrideKey, tc.override)
				require.Equal(t, http.StatusOK, code, out)
			}
			code, out := postSetting(t, h, instance.FaviconColorKey, "#0000ff")
			assert.Equal(t, tc.wantCode, code, out)
			if code == http.StatusOK {
				code, out = postSetting(t, h, instance.FaviconModeKey, instance.FaviconColor)
				require.Equal(t, http.StatusOK, code, out)
			}
			data := buildSettingsData(settingsBuildOpts{})
			assert.Equal(t, tc.wantSource, data.InstanceFaviconSource)
			assert.Equal(t, tc.wantLocked, data.InstanceFaviconLocked)
			assert.Equal(t, tc.wantOverride, data.InstanceFaviconOverride)
			assert.Equal(t, instance.FaviconColor, data.InstanceFaviconEffectiveMode)
			assert.Equal(t, tc.envColor, data.InstanceFaviconExternalColor)
			assert.Equal(t, tc.envBadge, data.InstanceFaviconExternalBadge)
			body := getFavicon(t, h, data.InstanceFaviconURL).Body.String()
			if tc.wantSource == instance.SourceEnv {
				assert.NotContains(t, body, "#0000ff", "the env value wins")
			} else {
				assert.Contains(t, body, `fill="#0000ff"`)
			}
		})
	}
}

func TestFaviconOverrideToggleKeepsSettings(t *testing.T) {
	h := setupFaviconAPI(t, "#ff0000", "")
	code, out := postSetting(t, h, instance.FaviconOverrideKey, "true")
	require.Equal(t, http.StatusOK, code, out)
	code, out = postSetting(t, h, instance.FaviconBadgeKey, "D")
	require.Equal(t, http.StatusOK, code, out)
	code, out = postSetting(t, h, instance.FaviconModeKey, instance.FaviconColor)
	require.Equal(t, http.StatusOK, code, out)
	overridden := out["favicon_url"]

	// Unchecking goes back to the env color and keeps the stored badge.
	code, out = postSetting(t, h, instance.FaviconOverrideKey, "false")
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, instance.SourceEnv, out["favicon_source"])
	assert.NotEqual(t, overridden, out["favicon_url"])
	assert.Equal(t, "D", db.GetSetting(instance.FaviconBadgeKey, ""))

	// Re-checking brings the stored favicon back with the same URL.
	code, out = postSetting(t, h, instance.FaviconOverrideKey, "true")
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, overridden, out["favicon_url"])
}

func TestFaviconURLChangesWithSetting(t *testing.T) {
	h := setupFaviconAPI(t, "", "")
	urls := map[string]bool{buildSettingsData(settingsBuildOpts{}).InstanceFaviconURL: true}
	for _, p := range [][2]string{
		{instance.FaviconModeKey, instance.FaviconColor},
		{instance.FaviconColorKey, "#112233"},
		{instance.FaviconBadgeKey, "X"},
		{instance.FaviconBadgeKey, "XY"},
	} {
		code, out := postSetting(t, h, p[0], p[1])
		require.Equal(t, http.StatusOK, code, out)
		u := buildSettingsData(settingsBuildOpts{}).InstanceFaviconURL
		assert.True(t, strings.HasPrefix(u, FaviconPath+"?v="), u)
		assert.False(t, urls[u], "%s=%s did not change the URL", p[0], p[1])
		urls[u] = true
	}
	// A stale ?v= and the bare /favicon.ico probe revalidate; a matching ETag is a 304.
	stale := getFavicon(t, h, FaviconPath+"?v=old")
	assert.Equal(t, "no-cache", stale.Header().Get("Cache-Control"))
	assert.Equal(t, http.StatusOK, getFavicon(t, h, "/favicon.ico").Code)
	req := httptest.NewRequest(http.MethodGet, FaviconPath, nil)
	req.Host = "127.0.0.1:7830"
	req.Header.Set("If-None-Match", stale.Header().Get("ETag"))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNotModified, rr.Code)
}

func TestGenericSettingsGetHidesUpload(t *testing.T) {
	h := setupFaviconAPI(t, "", "")
	pngURL, pngBytes := pngDataURL(t, 0)
	code, out := postSetting(t, h, instance.FaviconUploadKey, pngURL)
	require.Equal(t, http.StatusOK, code, out)
	assert.Equal(t, "", out["value"], "the response does not echo the upload")
	rr := getFavicon(t, h, "/api/settings")
	var settings map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &settings))
	assert.Equal(t, "", settings[instance.FaviconUploadKey])
	assert.Equal(t, instance.FaviconPNG, settings["instance_favicon_upload_type"])
	assert.Equal(t, len(pngBytes), mustAtoi(t, settings["instance_favicon_upload_bytes"]))
	assert.Equal(t, instance.FaviconUpload, settings["instance_favicon_effective_mode"])
}

func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	var n int
	require.NoError(t, json.Unmarshal([]byte(s), &n))
	return n
}
