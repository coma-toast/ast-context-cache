package embedder

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestSettingsForStoredProfile_onnx(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if err := db.SetSetting("embed_backend_profiles", `{"onnx":{"MODEL_DIR":"/tmp/onnx-model"}}`); err != nil {
		t.Fatal(err)
	}
	s := SettingsForStoredProfile("onnx")
	if s.ModelDir != "/tmp/onnx-model" {
		t.Fatalf("ModelDir = %q, want /tmp/onnx-model", s.ModelDir)
	}
	if s.Backend != "onnx" {
		t.Fatalf("Backend = %q, want onnx", s.Backend)
	}
}

func TestAuxBackend_defaultOnnx(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if got := AuxBackend(); got != "onnx" {
		t.Fatalf("AuxBackend() = %q, want onnx", got)
	}
}
