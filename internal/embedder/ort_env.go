package embedder

import (
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

var (
	errONNXRuntimeShutDown = errs.New("onnx runtime shut down")
	ortShutDown            atomic.Bool
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
	return sidecarLibPath(exePath + ".ortlib")
}

// sidecarLibPath returns the library path recorded in a sidecar file, or "" when the file is
// unreadable or names a library that doesn't exist (e.g. a build that resolved the wrong
// Homebrew), so resolution falls through to the default instead of failing to load.
func sidecarLibPath(sidecar string) string {
	data, err := os.ReadFile(sidecar)
	if err != nil {
		return ""
	}
	p := strings.TrimSpace(string(data))
	if p == "" {
		return ""
	}
	if _, err := os.Stat(p); err != nil {
		logger.Warn("Ignoring onnxruntime path from build sidecar", "sidecar", sidecar, "path", p, "error", err)
		return ""
	}
	return p
}

func ensureONNXRuntime() error {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(resolveORTLibPath())
		ortInitErr = ort.InitializeEnvironment()
	})
	if ortShutDown.Load() {
		return errONNXRuntimeShutDown
	}
	return ortInitErr
}

// destroyONNXRuntime tears down the ONNX Runtime environment if this process created
// one. Sessions must already be destroyed. The environment is not created again: later
// session creation fails, which is what a process on its way out wants.
func destroyONNXRuntime() {
	ortShutDown.Store(true)
	// Marks a never-initialized runtime done, so a racing ensureONNXRuntime can't create it.
	ortInitOnce.Do(func() { ortInitErr = errONNXRuntimeShutDown })
	if ortInitErr != nil || !ort.IsInitialized() {
		return
	}
	if err := ort.DestroyEnvironment(); err != nil {
		logger.Warn("Failed to destroy ONNX Runtime environment", "error", err)
	}
}
