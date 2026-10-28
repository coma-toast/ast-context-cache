package embedder

import (
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// RuntimeHooks runs after a successful embedder swap (tracked embedder is wired into consumers).
type RuntimeHooks struct {
	OnSwap func(tracked Interface)
}

var errRuntimeNotInitialized = errs.New("embedder runtime not initialized")

var (
	runtimeMu          sync.Mutex
	runtimeModelDir    string
	runtimeInitialized bool
	runtimeRaw         Interface
	runtimeTracked     Interface
	runtimeLoaded      func() bool
	runtimeHooks       RuntimeHooks
	onBeforeSwap       func()
)

// SetOnBeforeSwap registers a callback before tearing down the previous embedder (e.g. drain embed queue).
func SetOnBeforeSwap(fn func()) {
	onBeforeSwap = fn
}

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
		return errRuntimeNotInitialized
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
	if onBeforeSwap != nil {
		onBeforeSwap()
	}
	stopConnectivityProbe()
	runtimeMu.Lock()
	prevTracked := runtimeTracked
	runtimeRaw = nil
	runtimeTracked = nil
	runtimeMu.Unlock()
	shutdownEmbedder(prevTracked)
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
	FreezeWiredSnapshot()
	MarkReady()
	StartConnectivityProbe(raw)
	if runtimeHooks.OnSwap != nil {
		runtimeHooks.OnSwap(tracked)
	}
	wb, wm, _, _, wd := WiredSnapshot()
	logger.Info("Embedder active", "backend", wb, "model", wm, "dims", wd)
	return nil
}

// destroyRuntime is destroyONNXRuntime; a variable so tests keep the shared environment.
var destroyRuntime = destroyONNXRuntime

// Shutdown releases every native embedder resource before the process restarts or
// exits: the connectivity probe, the primary and aux embedders (their ONNX sessions and
// intra-op thread pools), and finally the ONNX Runtime environment. On macOS an execve
// can't complete while another thread sits in an uninterruptible kernel wait, so the
// restart path calls this before it execs. Embed calls afterwards fail instead of
// loading a new session.
func Shutdown() {
	stopConnectivityProbe()
	runtimeMu.Lock()
	prev := runtimeTracked
	runtimeRaw, runtimeTracked, runtimeLoaded = nil, nil, nil
	runtimeMu.Unlock()
	shutdownEmbedder(prev)
	stopAux()
	destroyRuntime()
	logger.Info("Embedder shut down")
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
