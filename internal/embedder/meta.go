package embedder

import "sync"

// Active* describe the live embedding provider metadata (set when wiring backends).
// All access must go through the functions below — activeMu guards every read
// and write, since SetActive can run (on a dashboard-triggered Reload) while
// per-request dimension checks (remote_http.go, remote_ollama.go,
// openai_embeddings.go) and the health-probe goroutine (health.go) are reading
// concurrently.
var (
	ActiveBackend  = "onnx"
	ActiveModel    = ModelName
	ActiveDim      = Dimensions
	ActiveRuntime  = "onnxruntime"
	ActiveEndpoint = ""
)

var (
	activeMu      sync.RWMutex
	wiredBackend  string
	wiredModel    string
	wiredDim      int
	wiredRuntime  string
	wiredEndpoint string
	wiredSet      bool
)

// ActiveSnapshot returns metadata for the currently wired embedder instance.
func ActiveSnapshot() (backend, model, runtime, endpoint string, dim int) {
	activeMu.RLock()
	defer activeMu.RUnlock()
	return ActiveBackend, ActiveModel, ActiveRuntime, ActiveEndpoint, ActiveDim
}

// WiredSnapshot returns metadata frozen at process start (dashboard "Active" row).
func WiredSnapshot() (backend, model, runtime, endpoint string, dim int) {
	activeMu.RLock()
	defer activeMu.RUnlock()
	if wiredSet {
		return wiredBackend, wiredModel, wiredRuntime, wiredEndpoint, wiredDim
	}
	return ActiveBackend, ActiveModel, ActiveRuntime, ActiveEndpoint, ActiveDim
}

// FreezeWiredSnapshot pins Active* as the running embedder for dashboard display (updated on Reload).
func FreezeWiredSnapshot() {
	activeMu.Lock()
	defer activeMu.Unlock()
	wiredBackend = ActiveBackend
	wiredModel = ActiveModel
	wiredDim = ActiveDim
	wiredRuntime = ActiveRuntime
	wiredEndpoint = ActiveEndpoint
	wiredSet = true
}

// SetActive updates metadata for /embed/health and logging. Call from main when wiring a backend.
func SetActive(backend, model string, dim int, runtime, endpoint string) {
	activeMu.Lock()
	defer activeMu.Unlock()
	ActiveBackend = backend
	if model != "" {
		ActiveModel = model
	}
	if dim > 0 {
		ActiveDim = dim
	}
	ActiveRuntime = runtime
	ActiveEndpoint = endpoint
}

// GetActiveDim safely reads the active embedding dimension. Prefer this over
// reading ActiveDim directly from outside this file.
func GetActiveDim() int {
	activeMu.RLock()
	defer activeMu.RUnlock()
	return ActiveDim
}

// GetActiveBackend safely reads the active backend name. Prefer this over
// reading ActiveBackend directly from outside this file.
func GetActiveBackend() string {
	activeMu.RLock()
	defer activeMu.RUnlock()
	return ActiveBackend
}
