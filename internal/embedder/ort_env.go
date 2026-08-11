package embedder

import (
	"os"
	"runtime"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func resolveORTLibPath() string {
	if p := os.Getenv("ONNXRUNTIME_LIB"); p != "" {
		return p
	}
	if p := ortLibFromSidecar(); p != "" {
		return p
	}
	if runtime.GOOS == "linux" {
		return "/usr/lib/libonnxruntime.so"
	}
	return "/opt/homebrew/lib/libonnxruntime.dylib"
}

// ortLibFromSidecar reads the "<binary>.ortlib" file `make build` writes next to the
// executable with the onnxruntime path it resolved for this machine, so a synced
// mcp-local config doesn't need a machine-specific ONNXRUNTIME_LIB override.
func ortLibFromSidecar() string {
	exePath, err := os.Executable()
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(exePath + ".ortlib")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func ensureONNXRuntime() error {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(resolveORTLibPath())
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}
