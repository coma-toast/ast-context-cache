package dashboard

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/ortinstall"
)

// No t.Parallel: the db pools, the access token and ortInstall are package globals.

// fakeORTInstall replaces the downloader; it records the data dirs it was called with.
func fakeORTInstall(t *testing.T, err error) *[]string {
	t.Helper()
	var calls []string
	prev := ortInstall
	t.Cleanup(func() { ortInstall = prev })
	ortInstall = func(ctx context.Context, dataDir string) (ortinstall.Installation, error) {
		calls = append(calls, dataDir)
		if err != nil {
			return ortinstall.Installation{}, err
		}
		dir := ortinstall.InstallDir(dataDir, ortinstall.Version)
		return ortinstall.Installation{Version: ortinstall.Version, Dir: dir, Lib: dir + "/libonnxruntime.so"}, nil
	}
	return &calls
}

func postORTInstall(t *testing.T, h http.Handler, method, remote string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, "/api/onnxruntime/install", nil)
	req.Host = "127.0.0.1:7830"
	if remote != "" {
		req.RemoteAddr = remote
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var out map[string]any
	json.Unmarshal(rr.Body.Bytes(), &out)
	return rr, out
}

func TestONNXRuntimeInstallEndpoint(t *testing.T) {
	h := setupInstanceAPI(t, "")
	calls := fakeORTInstall(t, nil)
	rr, out := postORTInstall(t, h, http.MethodPost, "")
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	assert.Equal(t, []string{db.GetDataDir()}, *calls)
	assert.Equal(t, "installed", out["status"])
	assert.Equal(t, ortinstall.Version, out["version"])
	assert.Equal(t, true, out["restart_required"])

	rr, _ = postORTInstall(t, h, http.MethodGet, "")
	assert.Equal(t, http.StatusMethodNotAllowed, rr.Code)
	assert.Len(t, *calls, 1)
}

func TestONNXRuntimeInstallEndpointErrors(t *testing.T) {
	h := setupInstanceAPI(t, "")
	tests := []struct {
		name string
		err  error
		code int
	}{
		{name: "checksum mismatch", err: errs.NewCode(errs.CodeConflict, "ONNX Runtime checksum mismatch"), code: http.StatusConflict},
		{name: "unsupported platform", err: errs.NewCode(errs.CodeUnsupported, "no managed ONNX Runtime build"), code: http.StatusBadRequest},
		{name: "download failure", err: errs.New("failed to download ONNX Runtime"), code: http.StatusBadGateway},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeORTInstall(t, tt.err)
			rr, out := postORTInstall(t, h, http.MethodPost, "")
			assert.Equal(t, tt.code, rr.Code)
			assert.Equal(t, tt.err.Error(), out["error"])
		})
	}
}

// TestONNXRuntimeInstallRequiresAuth: remote callers need the access token, like other mutating endpoints.
func TestONNXRuntimeInstallRequiresAuth(t *testing.T) {
	h := setupInstanceAPI(t, "")
	calls := fakeORTInstall(t, nil)
	httpguard.SetAccessToken("secret-token")
	t.Cleanup(func() { httpguard.SetAccessToken("") })
	rr, _ := postORTInstall(t, h, http.MethodPost, "100.64.0.9:5555")
	assert.Equal(t, http.StatusUnauthorized, rr.Code)
	assert.Empty(t, *calls)
}

func TestSettingsReportAuxAvailability(t *testing.T) {
	h := setupInstanceAPI(t, "")
	t.Setenv("ONNXRUNTIME_LIB", "/nonexistent/libonnxruntime.so")
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	req.Host = "127.0.0.1:7830"
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var settings map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &settings))
	for _, k := range []string{"aux_available", "aux_unavailable_reason", "onnxruntime_version", "onnxruntime_path"} {
		assert.Contains(t, settings, k)
	}
	data := buildSettingsData(settingsBuildOpts{})
	assert.Equal(t, data.ONNXRuntime.Available, data.AuxAvailable, "the default aux backend is onnx")
	assert.Equal(t, ortinstall.Version, data.ONNXRuntime.Pinned)
}
