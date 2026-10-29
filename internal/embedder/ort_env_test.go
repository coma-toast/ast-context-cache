package embedder

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/coma-toast/ast-context-cache/internal/ortinstall"
)

func TestEnsureONNXRuntimeIdempotent(t *testing.T) {
	err1 := ensureONNXRuntime()
	err2 := ensureONNXRuntime()
	if err1 != err2 {
		t.Fatalf("errs differ: %v vs %v", err1, err2)
	}
	if err1 != nil {
		t.Skipf("ONNX runtime unavailable in test env: %v", err1)
	}
}

func TestSidecarLibPath(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	lib := filepath.Join(dir, "libonnxruntime.dylib")
	if err := os.WriteFile(lib, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	write := func(name, content string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	tests := []struct {
		name    string
		sidecar string
		want    string
	}{
		{name: "existing library", sidecar: write("ok.ortlib", lib+"\n"), want: lib},
		{name: "library missing", sidecar: write("missing.ortlib", filepath.Join(dir, "nope", "libonnxruntime.dylib")), want: ""},
		{name: "empty sidecar", sidecar: write("empty.ortlib", "  \n"), want: ""},
		{name: "no sidecar", sidecar: filepath.Join(dir, "absent.ortlib"), want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, sidecarLibPath(tt.sidecar))
		})
	}
}

func TestResolveORTLibOrder(t *testing.T) {
	dataDir := t.TempDir()
	managed := filepath.Join(ortinstall.InstallDir(dataDir, "1.31.0"), ortinstall.LibName(runtime.GOOS))
	if err := os.MkdirAll(filepath.Dir(managed), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(managed, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	noSidecar := func() string { return "" }
	sidecar := func() string { return "/sidecar/libonnxruntime" }
	tests := []struct {
		name       string
		env        string
		sidecar    func() string
		dataDir    string
		wantPath   string
		wantSource string
	}{
		{name: "env wins", env: "/env/libonnxruntime", sidecar: sidecar, dataDir: dataDir, wantPath: "/env/libonnxruntime", wantSource: ORTSourceEnv},
		{name: "sidecar before managed", sidecar: sidecar, dataDir: dataDir, wantPath: "/sidecar/libonnxruntime", wantSource: ORTSourceSidecar},
		{name: "managed before default", sidecar: noSidecar, dataDir: dataDir, wantPath: managed, wantSource: ORTSourceManaged},
		{name: "platform default", sidecar: noSidecar, dataDir: t.TempDir(), wantSource: ORTSourceDefault},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("ONNXRUNTIME_LIB", tt.env)
			p, src := resolveORTLib(tt.dataDir, runtime.GOOS, tt.sidecar)
			assert.Equal(t, tt.wantSource, src)
			if tt.wantPath != "" {
				assert.Equal(t, tt.wantPath, p)
			}
		})
	}
}

func TestONNXRuntimeStatus(t *testing.T) {
	t.Parallel()
	dataDir := t.TempDir()
	missing := func() (string, string) { return filepath.Join(dataDir, "nope.so"), ORTSourceDefault }
	st := onnxRuntimeStatus(dataDir, nil, missing)
	assert.False(t, st.Available)
	assert.Contains(t, st.Reason, "not found")
	assert.False(t, st.Managed)
	assert.Equal(t, ortinstall.Version, st.Pinned)

	old := filepath.Join(ortinstall.InstallDir(dataDir, "1.25.0"), ortinstall.LibName(runtime.GOOS))
	if err := os.MkdirAll(filepath.Dir(old), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(old, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	managed := func() (string, string) { return old, ORTSourceManaged }
	st = onnxRuntimeStatus(dataDir, nil, managed)
	assert.True(t, st.Available)
	assert.True(t, st.Managed)
	assert.Equal(t, "1.25.0", st.Version)
	assert.Equal(t, ortinstall.Supported(), st.UpdateAvailable)

	failed := &ortLoadResult{path: "/usr/lib/libonnxruntime.so", source: ORTSourceDefault, err: os.ErrNotExist}
	st = onnxRuntimeStatus(dataDir, failed, managed)
	assert.False(t, st.Available)
	assert.True(t, st.Loaded)
	assert.True(t, st.RestartRequired, "a managed install exists that this process did not load")
	envFailed := &ortLoadResult{path: "/env/lib.so", source: ORTSourceEnv, err: os.ErrNotExist}
	assert.False(t, onnxRuntimeStatus(dataDir, envFailed, managed).RestartRequired, "ONNXRUNTIME_LIB still wins after a restart")
}
