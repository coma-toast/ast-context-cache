package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/ortinstall"
)

// TestMain keeps every CLI test off the network: `install --yes` would otherwise download ONNX Runtime.
func TestMain(m *testing.M) {
	ortInstall = func(ctx context.Context, dataDir string) (ortinstall.Installation, error) {
		return ortinstall.Installation{Version: ortinstall.Version, Dir: ortinstall.InstallDir(dataDir, ortinstall.Version)}, nil
	}
	ortStatus = func() embedder.ORTStatus {
		return embedder.ORTStatus{Available: true, Path: "/fake/libonnxruntime", Source: "default"}
	}
	os.Exit(m.Run())
}

// stubORT records install calls and fails them with installErr; the status reports unavailable.
func stubORT(t *testing.T, installErr error) *[]string {
	t.Helper()
	var calls []string
	prevInstall, prevStatus := ortInstall, ortStatus
	t.Cleanup(func() { ortInstall, ortStatus = prevInstall, prevStatus })
	ortInstall = func(ctx context.Context, dataDir string) (ortinstall.Installation, error) {
		calls = append(calls, dataDir)
		if installErr != nil {
			return ortinstall.Installation{}, installErr
		}
		return ortinstall.Installation{Version: ortinstall.Version, Dir: ortinstall.InstallDir(dataDir, ortinstall.Version)}, nil
	}
	ortStatus = func() embedder.ORTStatus {
		return embedder.ORTStatus{Reason: "ONNX Runtime library not found at /usr/lib/libonnxruntime.so (default)", Pinned: ortinstall.Version}
	}
	return &calls
}

func TestCLIInstallONNXRuntime(t *testing.T) {
	home := cliHome(t)
	calls := stubORT(t, nil)
	code, out, errOut := runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--yes")
	require.Equal(t, exitOK, code, errOut)
	require.Len(t, *calls, 1)
	assert.Equal(t, filepath.Join(home, ".astcache"), (*calls)[0])
	assert.Contains(t, out, "Installed ONNX Runtime "+ortinstall.Version)
	assert.Contains(t, errOut, "Installing ONNX Runtime")

	code, _, _ = runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--dry-run")
	assert.Equal(t, exitOK, code)
	code, _, _ = runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--yes", "--skip-onnxruntime")
	assert.Equal(t, exitOK, code)
	assert.Len(t, *calls, 1, "dry-run and --skip-onnxruntime don't install")
}

func TestCLIInstallSkipsInstalledONNXRuntime(t *testing.T) {
	home := cliHome(t)
	calls := stubORT(t, nil)
	dir := ortinstall.InstallDir(filepath.Join(home, ".astcache"), ortinstall.Version)
	require.NoError(t, os.MkdirAll(dir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, ortinstall.LibName(runtime.GOOS)), nil, 0o644))
	code, out, errOut := runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--yes")
	require.Equal(t, exitOK, code, errOut)
	assert.Empty(t, *calls)
	assert.Contains(t, out, "already installed")
}

// TestCLIInstallContinuesOnONNXRuntimeFailure: a failed download or checksum is a warning.
func TestCLIInstallContinuesOnONNXRuntimeFailure(t *testing.T) {
	home := cliHome(t)
	stubORT(t, errs.NewCode(errs.CodeConflict, "ONNX Runtime checksum mismatch"))
	code, out, errOut := runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--yes", "--json")
	require.Equal(t, exitOK, code, errOut)
	var top map[string]json.RawMessage
	require.NoError(t, json.Unmarshal([]byte(out), &top))
	assert.Equal(t, []string{"changes", "status", "warnings"}, sortedKeys(top), "the --json shape is unchanged")
	var res cliOutput
	require.NoError(t, json.Unmarshal([]byte(out), &res))
	assert.Contains(t, strings.Join(res.Warnings, "\n"), "aux embedder unavailable: ONNX Runtime checksum mismatch")
	assert.Equal(t, "installed", res.Status[0].Status)
	assert.Contains(t, readTestFile(t, filepath.Join(home, ".cursor", "mcp.json")), "/mcp")
}

func TestCLIVerifyReportsONNXRuntime(t *testing.T) {
	cliHome(t)
	stubORT(t, nil)
	code, out, _ := runCLITest(t, "verify", "--target", "cursor")
	assert.Equal(t, exitOK, code)
	assert.Contains(t, out, "ONNX Runtime: missing (ONNX Runtime library not found at /usr/lib/libonnxruntime.so (default))")
	code, out, _ = runCLITest(t, "verify", "--target", "cursor", "--json")
	assert.Equal(t, exitOK, code)
	var res cliOutput
	require.NoError(t, json.Unmarshal([]byte(out), &res))
	assert.Contains(t, strings.Join(res.Warnings, "\n"), "aux embedder unavailable")
	assert.Equal(t, "ONNX Runtime: 1.31.0 at /x/libonnxruntime.so (managed); 1.32.0 available via `ast-mcp install --target <host> --yes`",
		ortStatusLine(embedder.ORTStatus{Available: true, Version: "1.31.0", Path: "/x/libonnxruntime.so", Source: "managed", Pinned: "1.32.0", UpdateAvailable: true}))
}

func TestAddORTHealth(t *testing.T) {
	missing := embedder.ORTStatus{Reason: "ONNX Runtime library not found", Path: "/usr/lib/libonnxruntime.so", Source: "default"}
	p := map[string]interface{}{}
	addORTHealth(p, missing, "onnx")
	assert.Equal(t, false, p["aux_available"])
	assert.Equal(t, "ONNX Runtime library not found", p["aux_unavailable_reason"])
	assert.Equal(t, "/usr/lib/libonnxruntime.so", p["onnxruntime_path"])
	addORTHealth(p, missing, "ollama")
	assert.Equal(t, true, p["aux_available"], "a remote aux backend needs no ONNX Runtime")
	assert.Equal(t, "", p["aux_unavailable_reason"])
	addORTHealth(p, embedder.ORTStatus{Available: true, Version: "1.31.0", Path: "/d/lib/onnxruntime-1.31.0/libonnxruntime.so", Source: "managed"}, "onnx")
	assert.Equal(t, true, p["aux_available"])
	assert.Equal(t, "1.31.0", p["onnxruntime_version"])
}
