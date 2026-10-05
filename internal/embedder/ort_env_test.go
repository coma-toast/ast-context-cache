package embedder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
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
