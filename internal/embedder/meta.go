package embedder

// Active* describe the live embedding provider metadata (set when wiring backends).
var (
	ActiveBackend  = "onnx"
	ActiveModel    = ModelName
	ActiveDim      = Dimensions
	ActiveRuntime  = "onnxruntime"
	ActiveEndpoint = ""
)

var (
	wiredBackend  string
	wiredModel    string
	wiredDim      int
	wiredRuntime  string
	wiredEndpoint string
	wiredSet      bool
)

// ActiveSnapshot returns metadata for the currently wired embedder instance.
func ActiveSnapshot() (backend, model, runtime, endpoint string, dim int) {
	return ActiveBackend, ActiveModel, ActiveRuntime, ActiveEndpoint, ActiveDim
}

// WiredSnapshot returns metadata frozen at process start (dashboard "Active" row).
func WiredSnapshot() (backend, model, runtime, endpoint string, dim int) {
	if wiredSet {
		return wiredBackend, wiredModel, wiredRuntime, wiredEndpoint, wiredDim
	}
	return ActiveSnapshot()
}

// FreezeWiredSnapshot pins Active* as the running embedder for dashboard display (updated on Reload).
func FreezeWiredSnapshot() {
	wiredBackend = ActiveBackend
	wiredModel = ActiveModel
	wiredDim = ActiveDim
	wiredRuntime = ActiveRuntime
	wiredEndpoint = ActiveEndpoint
	wiredSet = true
}

// SetActive updates metadata for /embed/health and logging. Call from main when wiring a backend.
func SetActive(backend, model string, dim int, runtime, endpoint string) {
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
