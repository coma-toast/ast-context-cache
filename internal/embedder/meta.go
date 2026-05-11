package embedder

// Active* describe the running embedding provider (set at process start from NewForMain).
var (
	ActiveBackend  = "onnx"
	ActiveModel    = ModelName
	ActiveDim      = Dimensions
	ActiveRuntime  = "onnxruntime"
	ActiveEndpoint = ""
)

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
