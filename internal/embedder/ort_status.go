package embedder

import (
	"os"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/ortinstall"
)

// ortLoadResult is what ensureONNXRuntime saw when it loaded the library.
type ortLoadResult struct {
	path, source, version string
	err                   error
}

var ortLoaded atomic.Pointer[ortLoadResult]

// ORTStatus describes whether ONNX Runtime is usable by the local (onnx) embedder.
type ORTStatus struct {
	// Available is true when the runtime loaded, or (before it is loaded) its library exists.
	Available bool   `json:"available"`
	Reason    string `json:"reason,omitempty"`
	// Loaded is true once this process has tried to load the runtime.
	Loaded  bool   `json:"loaded"`
	Version string `json:"version,omitempty"`
	Path    string `json:"path"`
	Source  string `json:"source"`
	Managed bool   `json:"managed"`
	// Installed is the newest managed install on disk, if any.
	Installed string `json:"installed,omitempty"`
	// Pinned is the version ast-mcp installs; UpdateAvailable is true when the managed install is older.
	Pinned          string `json:"pinned"`
	UpdateAvailable bool   `json:"update_available"`
	// RestartRequired is true when a managed install on disk is not the library this process loaded.
	RestartRequired  bool `json:"restart_required"`
	InstallSupported bool `json:"install_supported"`
}

// ONNXRuntimeStatus reports the ONNX Runtime state without loading it: the load result once
// the runtime has been initialized, otherwise whether the library it would load exists.
func ONNXRuntimeStatus() ORTStatus {
	return onnxRuntimeStatus(db.GetDataDir(), ortLoaded.Load(), ResolveORTLib)
}

func onnxRuntimeStatus(dataDir string, r *ortLoadResult, resolve func() (string, string)) ORTStatus {
	st := ORTStatus{Pinned: ortinstall.Version, InstallSupported: ortinstall.Supported()}
	inst, hasManaged := ortinstall.Installed(dataDir)
	if hasManaged {
		st.Installed = inst.Version
	}
	if r != nil {
		st.Path, st.Source, st.Loaded = r.path, r.source, true
		if r.err != nil {
			st.Reason = r.err.Error()
		} else {
			st.Available, st.Version = true, r.version
		}
		// Only managed and default resolve to the managed install after a restart; env and sidecar win.
		st.RestartRequired = hasManaged && inst.Lib != r.path && (r.source == ORTSourceManaged || r.source == ORTSourceDefault)
	} else {
		st.Path, st.Source = resolve()
		if _, err := os.Stat(st.Path); err != nil {
			st.Reason = "ONNX Runtime library not found at " + st.Path + " (" + st.Source + ")"
		} else {
			st.Available = true
		}
	}
	st.Managed = st.Source == ORTSourceManaged
	if st.Managed && st.Version == "" {
		st.Version = ortinstall.VersionFromLibPath(dataDir, st.Path)
	}
	st.UpdateAvailable = st.InstallSupported && (st.Managed || hasManaged) && ortinstall.UpdateAvailable(st.Installed)
	return st
}

// AuxAvailable reports whether the aux (catch-up) embedder can run, with the reason when it
// can't. Only the onnx aux backend needs ONNX Runtime.
func AuxAvailable() (bool, string) {
	return AuxAvailableFor(AuxBackend(), ONNXRuntimeStatus())
}

// AuxAvailableFor is AuxAvailable for a given aux backend and ONNX Runtime status.
func AuxAvailableFor(auxBackend string, st ORTStatus) (bool, string) {
	if normalizeEmbedBackend(auxBackend) != "onnx" {
		return true, ""
	}
	return st.Available, st.Reason
}
