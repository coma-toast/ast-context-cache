package embedder

import (
	"fmt"
	"log"
	"sync"
)

// RuntimeHooks runs after a successful embedder swap (tracked embedder is wired into consumers).
type RuntimeHooks struct {
	OnSwap func(tracked Interface)
}

var (
	runtimeMu           sync.Mutex
	runtimeModelDir     string
	runtimeInitialized  bool
	runtimeRaw          Interface
	runtimeTracked   Interface
	runtimeLoaded    func() bool
	runtimeHooks     RuntimeHooks
)

// SetRuntimeHooks registers callbacks invoked after InitRuntime or Reload swaps embedders.
func SetRuntimeHooks(h RuntimeHooks) {
	runtimeHooks = h
}

// InitRuntime wires the process embedder from effective settings (env + DB).
func InitRuntime(modelDir string) error {
	runtimeModelDir = modelDir
	runtimeInitialized = true
	return reloadRuntime()
}

// Reload rebuilds the embedder from current effective settings and swaps it in without restarting ast-mcp.
func Reload() error {
	if !runtimeInitialized {
		return fmt.Errorf("embedder runtime not initialized")
	}
	return reloadRuntime()
}

// Tracked returns the health-wrapped embedder used by MCP, queue, and search.
func Tracked() Interface {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	return runtimeTracked
}

// Raw returns the underlying embedder (for connectivity probes).
func Raw() Interface {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	return runtimeRaw
}

// IsLoaded reports whether a lazy ONNX embedder has loaded its model.
func IsLoaded() bool {
	runtimeMu.Lock()
	fn := runtimeLoaded
	runtimeMu.Unlock()
	if fn == nil {
		return false
	}
	return fn()
}

func reloadRuntime() error {
	runtimeMu.Lock()
	prevTracked := runtimeTracked
	runtimeMu.Unlock()
	raw, loaded, err := NewForMain(runtimeModelDir)
	if err != nil {
		return err
	}
	tracked := TrackHealth(raw)
	runtimeMu.Lock()
	runtimeRaw = raw
	runtimeTracked = tracked
	runtimeLoaded = loaded
	runtimeMu.Unlock()
	shutdownEmbedder(prevTracked)
	FreezeWiredSnapshot()
	MarkReady()
	stopConnectivityProbe()
	StartConnectivityProbe(raw)
	if runtimeHooks.OnSwap != nil {
		runtimeHooks.OnSwap(tracked)
	}
	wb, wm, _, _, wd := WiredSnapshot()
	log.Printf("Embedder active: backend=%s model=%s dims=%d", wb, wm, wd)
	return nil
}

func shutdownEmbedder(e Interface) {
	if e == nil {
		return
	}
	if t, ok := e.(*healthTracker); ok {
		shutdownEmbedder(t.inner)
		return
	}
	if s, ok := e.(interface{ Stop() }); ok {
		s.Stop()
	}
}
