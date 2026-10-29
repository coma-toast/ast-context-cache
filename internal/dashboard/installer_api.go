package dashboard

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/installer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// installerUnavailableMsg answers installer requests made before main has set the service (the
// database is still opening, or it failed to open).
const installerUnavailableMsg = "installer not ready; the database is still opening"

var (
	installerMu  sync.RWMutex
	installerSvc installer.Service

	// installerRecheckDelays are when, after an apply, components written while their host was
	// running are re-verified. Claude Desktop has been seen dropping an entry 30s after the write.
	installerRecheckDelays = []time.Duration{10 * time.Second, 30 * time.Second, 60 * time.Second}
)

// installerComponent is one target × component cell: where it is written plus its on-disk status.
type installerComponent struct {
	installer.ComponentInfo
	Status       installer.Status `json:"status"`
	StatusReason string           `json:"status_reason,omitempty"`
}

// installerTarget is one host card in Settings → Agent integration.
type installerTarget struct {
	ID         installer.Target     `json:"id"`
	Name       string               `json:"name"`
	Components []installerComponent `json:"components"`
}

// installerOverview is the Agent integration card data. HooksEnabled shows the Hooks row: on
// while feature_handoff_hooks or feature_search_nudge is.
type installerOverview struct {
	Targets        []installerTarget `json:"targets"`
	LegacyWarnings []string          `json:"legacy_warnings"`
	HooksEnabled   bool              `json:"hooks_enabled"`
}

type installerApplyRequest struct {
	PlanID string `json:"plan_id"`
}

type installerRestoreRequest struct {
	BackupID string `json:"backup_id"`
}

// installerErrorBody is every installer error response. Repreview tells the UI the plan is gone
// or stale (file changed, expired, unknown) and a fresh preview replaces it (IN-5).
type installerErrorBody struct {
	Error     string    `json:"error"`
	Code      errs.Code `json:"code,omitempty"`
	Repreview bool      `json:"repreview,omitempty"`
}

// SetInstaller gives the dashboard the installer service it drives. main calls it once the
// database is open, so the service's one-time legacy install-record check (IN-13) can run.
func SetInstaller(svc installer.Service) {
	installerMu.Lock()
	defer installerMu.Unlock()
	installerSvc = svc
}

func registerInstallerAPI(mux *http.ServeMux) {
	mux.HandleFunc("/api/dashboard/installer", handleInstallerOverview)
	mux.HandleFunc("/api/dashboard/installer/preview", handleInstallerPreview)
	mux.HandleFunc("/api/dashboard/installer/apply", handleInstallerApply)
	mux.HandleFunc("/api/dashboard/installer/backups", handleInstallerBackups)
	mux.HandleFunc("/api/dashboard/installer/restore", handleInstallerRestore)
	mux.HandleFunc("/api/dashboard/installer/legacy/dismiss", handleInstallerLegacyDismiss)
}

// handleInstallerOverview lists every target's components with their status from disk (IN-8),
// plus the legacy install warnings (IN-13).
func handleInstallerOverview(w http.ResponseWriter, r *http.Request) {
	svc, ok := installerFor(w, r, http.MethodGet)
	if !ok {
		return
	}
	statuses, err := svc.Verify(nil)
	if err != nil {
		writeInstallerError(w, err)
		return
	}
	legacy := svc.LegacyWarnings()
	if len(legacy) > 0 {
		// Showing them here is their one-time display: later CLI installs stay quiet, while
		// this card keeps them until dismissed.
		svc.ClaimLegacyWarnings()
	}
	writeInstallerJSON(w, http.StatusOK, installerOverview{
		Targets:        mergeInstallerStatus(svc.Targets(), statuses),
		LegacyWarnings: legacy,
		HooksEnabled:   flags.Enabled(flags.KeyHandoffHooks) || flags.Enabled(flags.KeySearchNudge),
	})
}

// handleInstallerLegacyDismiss hides the pre-4.0 install warnings everywhere.
func handleInstallerLegacyDismiss(w http.ResponseWriter, r *http.Request) {
	svc, ok := installerFor(w, r, http.MethodPost)
	if !ok {
		return
	}
	if err := svc.DismissLegacyWarnings(); err != nil {
		writeInstallerError(w, err)
		return
	}
	realtime.Notify(realtime.SettingsChanged)
	writeInstallerJSON(w, http.StatusOK, map[string]string{"status": "dismissed"})
}

// handleInstallerPreview plans an install or uninstall and returns the per-file diff. Nothing is
// written until the plan is applied.
func handleInstallerPreview(w http.ResponseWriter, r *http.Request) {
	svc, ok := installerFor(w, r, http.MethodPost)
	if !ok {
		return
	}
	var req installer.PlanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeInstallerJSON(w, http.StatusBadRequest, installerErrorBody{Error: "invalid JSON body", Code: errs.CodeInvalidInput})
		return
	}
	plan, err := svc.Plan(req)
	if err != nil {
		writeInstallerError(w, err)
		return
	}
	writeInstallerJSON(w, http.StatusOK, plan)
}

// handleInstallerApply writes a previewed plan. A plan whose files changed, that expired, or
// that is unknown answers 409 with repreview set, and nothing is written.
func handleInstallerApply(w http.ResponseWriter, r *http.Request) {
	svc, ok := installerFor(w, r, http.MethodPost)
	if !ok {
		return
	}
	var req installerApplyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.PlanID == "" {
		writeInstallerJSON(w, http.StatusBadRequest, installerErrorBody{Error: "plan_id required", Code: errs.CodeInvalidInput})
		return
	}
	res, err := svc.Apply(req.PlanID)
	if res != nil && len(res.Written) > 0 {
		// A partial apply still changed files, so listeners refresh even on error.
		realtime.Notify(realtime.SettingsChanged)
	}
	if err != nil {
		writeApplyError(w, err)
		return
	}
	if len(res.Recheck) > 0 {
		go recheckInstall(svc, res.Recheck, installerRecheckDelays)
	}
	writeInstallerJSON(w, http.StatusOK, res)
}

// recheckInstall re-verifies components written while their host was running, stopping at the
// first change: Recheck logs it, and listeners refresh to show the new status and its reason.
func recheckInstall(svc installer.Service, applied []installer.ComponentStatus, delays []time.Duration) {
	start := time.Now()
	for _, d := range delays {
		time.Sleep(time.Until(start.Add(d)))
		_, warnings, err := svc.Recheck(applied)
		if err != nil {
			logger.Warn("Installer recheck failed", "error", err)
			return
		}
		if len(warnings) > 0 {
			realtime.Notify(realtime.SettingsChanged)
			return
		}
	}
}

// handleInstallerBackups lists saved backups, newest first.
func handleInstallerBackups(w http.ResponseWriter, r *http.Request) {
	svc, ok := installerFor(w, r, http.MethodGet)
	if !ok {
		return
	}
	backups, err := svc.Backups()
	if err != nil {
		writeInstallerError(w, err)
		return
	}
	writeInstallerJSON(w, http.StatusOK, map[string]any{"backups": backups})
}

// handleInstallerRestore writes a backup back to its original path.
func handleInstallerRestore(w http.ResponseWriter, r *http.Request) {
	svc, ok := installerFor(w, r, http.MethodPost)
	if !ok {
		return
	}
	var req installerRestoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.BackupID == "" {
		writeInstallerJSON(w, http.StatusBadRequest, installerErrorBody{Error: "backup_id required", Code: errs.CodeInvalidInput})
		return
	}
	if err := svc.Restore(req.BackupID); err != nil {
		writeInstallerError(w, err)
		return
	}
	realtime.Notify(realtime.SettingsChanged)
	writeInstallerJSON(w, http.StatusOK, map[string]string{"status": "restored", "backup_id": req.BackupID})
}

// installerFor checks the method and returns the service, writing the error response itself
// when either is missing.
func installerFor(w http.ResponseWriter, r *http.Request, method string) (installer.Service, bool) {
	if r.Method != method {
		writeInstallerJSON(w, http.StatusMethodNotAllowed, installerErrorBody{Error: method + " required"})
		return nil, false
	}
	installerMu.RLock()
	svc := installerSvc
	installerMu.RUnlock()
	if svc == nil {
		writeInstallerJSON(w, http.StatusServiceUnavailable, installerErrorBody{Error: installerUnavailableMsg})
		return nil, false
	}
	return svc, true
}

// mergeInstallerStatus pairs each target's components with their verified status.
func mergeInstallerStatus(targets []installer.TargetInfo, statuses []installer.ComponentStatus) []installerTarget {
	type key struct {
		target    installer.Target
		component installer.Component
	}
	byKey := make(map[key]installer.ComponentStatus, len(statuses))
	for _, st := range statuses {
		byKey[key{st.Target, st.Component}] = st
	}
	out := make([]installerTarget, 0, len(targets))
	for _, t := range targets {
		it := installerTarget{ID: t.ID, Name: t.Name, Components: make([]installerComponent, 0, len(t.Components))}
		for _, c := range t.Components {
			st, ok := byKey[key{t.ID, c.Component}]
			if !ok {
				st.Status = installer.StatusNotInstalled
			}
			it.Components = append(it.Components, installerComponent{ComponentInfo: c, Status: st.Status, StatusReason: st.Reason})
		}
		out = append(out, it)
	}
	return out
}

// writeInstallerError maps an installer error to its HTTP status: invalid input 400, unknown
// backup 404, a conflict 409, anything else 500.
func writeInstallerError(w http.ResponseWriter, err error) {
	body := installerErrorBody{Error: err.Error(), Code: errs.CodeOf(err)}
	status := http.StatusInternalServerError
	switch {
	case errs.HasCode(err, errs.CodeInvalidInput):
		status = http.StatusBadRequest
	case errs.HasCode(err, errs.CodeNotFound):
		status = http.StatusNotFound
	case errs.HasCode(err, errs.CodeConflict):
		status = http.StatusConflict
	default:
		logger.Warn("Installer request failed", "error", err)
	}
	writeInstallerJSON(w, status, body)
}

// writeApplyError answers 409 with repreview for a plan that is stale, expired, or unknown:
// Apply wrote nothing and a fresh preview replaces it (IN-5).
func writeApplyError(w http.ResponseWriter, err error) {
	if !isRepreviewError(err) {
		writeInstallerError(w, err)
		return
	}
	writeInstallerJSON(w, http.StatusConflict, installerErrorBody{Error: err.Error(), Code: errs.CodeOf(err), Repreview: true})
}

func isRepreviewError(err error) bool {
	return errs.HasCode(err, errs.CodeConflict) || errs.HasCode(err, errs.CodeExpired) || errs.HasCode(err, errs.CodeNotFound)
}

func writeInstallerJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
