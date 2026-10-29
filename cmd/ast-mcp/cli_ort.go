package main

import (
	"context"
	"fmt"
	"io"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/ortinstall"
)

// ortInstall runs the verified managed ONNX Runtime install. Tests replace it.
var ortInstall = func(ctx context.Context, dataDir string) (ortinstall.Installation, error) {
	return (&ortinstall.Installer{DataDir: dataDir}).Install(ctx)
}

// ortStatus reports the ONNX Runtime the server would load. Tests replace it.
var ortStatus = embedder.ONNXRuntimeStatus

// installONNXRuntime installs the pinned ONNX Runtime unless a managed install of it (or newer)
// exists. A failure never fails the install: it comes back as a warning, which says "aux embedder
// unavailable" when no usable runtime is left.
func installONNXRuntime(ctx context.Context, dataDir string, progress io.Writer) (note string, warnings []string) {
	if inst, ok := ortinstall.Installed(dataDir); ok && ortinstall.CompareVersions(inst.Version, ortinstall.Version) >= 0 {
		return "ONNX Runtime " + inst.Version + " already installed at " + inst.Dir, nil
	}
	fmt.Fprintf(progress, "Installing ONNX Runtime %s into %s (skip with --skip-onnxruntime)…\n", ortinstall.Version, ortinstall.LibRoot(dataDir))
	inst, err := ortInstall(ctx, dataDir)
	if err == nil {
		return "Installed ONNX Runtime " + inst.Version + " at " + inst.Dir + "; restart ast-mcp to load it", nil
	}
	logger.Warn("ONNX Runtime install failed", "data_dir", dataDir, "error", err)
	if st := ortStatus(); st.Available {
		return "", []string{"onnxruntime install failed: " + err.Error() + " (still using " + st.Path + ")"}
	}
	return "", []string{"aux embedder unavailable: " + err.Error()}
}

// ortStatusLine renders the ONNX Runtime state for `ast-mcp verify`.
func ortStatusLine(st embedder.ORTStatus) string {
	if !st.Available {
		return "ONNX Runtime: missing (" + st.Reason + "); aux embedder unavailable. Run `ast-mcp install --target <host> --yes` to install " + st.Pinned
	}
	ver := st.Version
	if ver == "" {
		ver = "version unknown"
	}
	line := "ONNX Runtime: " + ver + " at " + st.Path + " (" + st.Source + ")"
	if st.UpdateAvailable {
		line += "; " + st.Pinned + " available via `ast-mcp install --target <host> --yes`"
	}
	return line
}
