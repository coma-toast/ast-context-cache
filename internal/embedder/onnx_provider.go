package embedder

import (
	"path/filepath"
	"runtime"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// onnxProviderSetting selects the ONNX Runtime execution provider for the local embedder.
const onnxProviderSetting = "EMBED_ONNX_PROVIDER"

const (
	onnxProviderAuto   = "auto"
	onnxProviderCPU    = "cpu"
	onnxProviderCoreML = "coreml"
	onnxProviderCUDA   = "cuda"
)

// resolveONNXProvider maps a setting value to a concrete provider for goos.
// auto (or empty) uses CoreML on macOS and the CPU elsewhere; unknown values fall back to the CPU.
func resolveONNXProvider(value, goos string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case onnxProviderCPU:
		return onnxProviderCPU
	case onnxProviderCoreML:
		return onnxProviderCoreML
	case onnxProviderCUDA:
		return onnxProviderCUDA
	case "", onnxProviderAuto:
		if goos == "darwin" {
			return onnxProviderCoreML
		}
		return onnxProviderCPU
	default:
		logger.Warn("Unknown EMBED_ONNX_PROVIDER; using the CPU", "value", value)
		return onnxProviderCPU
	}
}

// ONNXProvider returns the execution provider the local embedder asks for.
func ONNXProvider() string {
	return resolveONNXProvider(EffectiveEnv(onnxProviderSetting), runtime.GOOS)
}

// newONNXSessionOptions builds session options that append provider. The CPU needs none (nil).
func newONNXSessionOptions(provider, cacheDir string) (*ort.SessionOptions, error) {
	if provider == onnxProviderCPU {
		return nil, nil
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, errs.WrapMessage("failed to create ONNX session options", err)
	}
	switch provider {
	case onnxProviderCoreML:
		// With the fixed-shape model (see static_model.go) CoreML compiles the whole graph; MLProgram
		// was 2x faster than the CPU and ~3x faster than NeuralNetwork on an M4.
		coreml := map[string]string{
			"ModelFormat":              "MLProgram",
			"MLComputeUnits":           "ALL",
			"RequireStaticInputShapes": "1",
		}
		// Compiling the model takes about 2 minutes on an M4; the cache makes later starts fast.
		if cacheDir != "" {
			coreml["ModelCacheDirectory"] = cacheDir
		}
		err = opts.AppendExecutionProviderCoreMLV2(coreml)
	case onnxProviderCUDA:
		var cuda *ort.CUDAProviderOptions
		cuda, err = ort.NewCUDAProviderOptions()
		if err == nil {
			defer cuda.Destroy()
			err = opts.AppendExecutionProviderCUDA(cuda)
		}
	}
	if err != nil {
		opts.Destroy()
		return nil, errs.WrapMessage("failed to enable ONNX execution provider", err, "provider", provider)
	}
	return opts, nil
}

// newONNXSession opens the embedding model with provider's session options.
func newONNXSession(modelPath, provider string, inputs, outputs []ort.ArbitraryTensor) (*ort.AdvancedSession, error) {
	opts, err := newONNXSessionOptions(provider, filepath.Join(filepath.Dir(modelPath), "coreml-cache"))
	if err != nil {
		return nil, err
	}
	if opts != nil {
		defer opts.Destroy()
	}
	return ort.NewAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"sentence_embedding"},
		inputs, outputs, opts)
}

// Provider reports the execution provider the session runs on (cpu, coreml or cuda).
func (e *Embedder) Provider() string {
	if e == nil {
		return ""
	}
	return e.provider
}
