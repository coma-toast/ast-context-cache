package embedder

import (
	"os"
	"runtime"
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
	if runtime.GOOS == "linux" {
		return "/usr/lib/libonnxruntime.so"
	}
	return "/opt/homebrew/lib/libonnxruntime.dylib"
}

func ensureONNXRuntime() error {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(resolveORTLibPath())
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}
