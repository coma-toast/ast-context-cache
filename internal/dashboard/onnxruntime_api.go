package dashboard

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/ortinstall"
)

const ortInstallTimeout = 15 * time.Minute

// ortInstall runs the verified managed ONNX Runtime install. Tests replace it with a fake.
var ortInstall = func(ctx context.Context, dataDir string) (ortinstall.Installation, error) {
	return (&ortinstall.Installer{DataDir: dataDir}).Install(ctx)
}

// populateORTData fills the ONNX Runtime and aux embedder availability for Settings.
func populateORTData(data *components.SettingsData) {
	st := embedder.ONNXRuntimeStatus()
	data.AuxAvailable, data.AuxUnavailableReason = embedder.AuxAvailableFor(embedder.AuxBackend(), st)
	data.ONNXRuntime = st
}

// applyORTSettings adds the read-only ONNX Runtime keys to the /api/settings map.
func applyORTSettings(settings map[string]string) {
	var data components.SettingsData
	populateORTData(&data)
	settings["aux_available"] = strconv.FormatBool(data.AuxAvailable)
	settings["aux_unavailable_reason"] = data.AuxUnavailableReason
	settings["onnxruntime_version"] = data.ONNXRuntime.Version
	settings["onnxruntime_path"] = data.ONNXRuntime.Path
}

// handleONNXRuntimeInstall runs the same verified install as `ast-mcp install` (POST). The
// server loads the new runtime after a restart (/api/restart-now).
func handleONNXRuntimeInstall(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "POST required"})
		return
	}
	// Detached from the request so a closed tab doesn't abort a half-written install.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), ortInstallTimeout)
	defer cancel()
	inst, err := ortInstall(ctx, db.GetDataDir())
	if err != nil {
		logger.Warn("ONNX Runtime install failed", "error", err)
		w.WriteHeader(ortInstallStatus(err))
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	logger.Info("ONNX Runtime installed from dashboard", "version", inst.Version, "path", inst.Dir)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           "installed",
		"version":          inst.Version,
		"path":             inst.Lib,
		"restart_required": true,
		"onnxruntime":      embedder.ONNXRuntimeStatus(),
	})
}

func ortInstallStatus(err error) int {
	switch {
	case errs.HasCode(err, errs.CodeUnsupported), errs.HasCode(err, errs.CodeInvalidInput):
		return http.StatusBadRequest
	case errs.HasCode(err, errs.CodeConflict):
		return http.StatusConflict
	}
	return http.StatusBadGateway
}
