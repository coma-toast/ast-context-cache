package dashboard

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// flagSettingKeyMsg is the generic settings POST's answer for a flag key, so feature flags have
// one write path, which validates the key and honors env locks.
const flagSettingKeyMsg = "use /api/dashboard/flags for feature flags"

type setFlagRequest struct {
	Key     string `json:"key"`
	Enabled *bool  `json:"enabled"`
}

// handleDashboardFlagsJSON lists every flag's resolved state (GET) or sets one flag (POST
// {"key", "enabled"}). An env-locked flag answers 409 and an unknown key 404.
func handleDashboardFlagsJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		json.NewEncoder(w).Encode(map[string]any{"flags": flags.State()})
	case http.MethodPost:
		postFlag(w, r)
	default:
		writeFlagsError(w, http.StatusMethodNotAllowed, "GET or POST required")
	}
}

func flagEnabledBefore(key string) (bool, bool) {
	for _, st := range flags.State() {
		if st.Key == key {
			return st.Enabled, true
		}
	}
	return false, false
}

func postFlag(w http.ResponseWriter, r *http.Request) {
	var req setFlagRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeFlagsError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Key == "" || req.Enabled == nil {
		writeFlagsError(w, http.StatusBadRequest, "key and enabled required")
		return
	}
	prev, hasPrev := flagEnabledBefore(req.Key)
	if err := flags.Set(req.Key, *req.Enabled); err != nil {
		writeFlagsError(w, flagErrorStatus(err), err.Error())
		return
	}
	notify := realtime.SettingsChanged
	if hasPrev && prev != *req.Enabled {
		if recordSettingsConfigChange(req.Key, strconv.FormatBool(prev), strconv.FormatBool(*req.Enabled), "Feature flag toggled") {
			notify |= realtime.KnowledgeBase
		}
	}
	realtime.Notify(notify)
	json.NewEncoder(w).Encode(map[string]any{"status": "ok", "flags": flags.State()})
}

// isFlagKey reports whether key is a registered feature flag (flag keys double as settings keys).
func isFlagKey(key string) bool {
	for _, f := range flags.All() {
		if f.Key == key {
			return true
		}
	}
	return false
}

func flagErrorStatus(err error) int {
	switch {
	case errs.HasCode(err, errs.CodeConflict):
		return http.StatusConflict
	case errs.HasCode(err, errs.CodeNotFound):
		return http.StatusNotFound
	default:
		logger.Warn("Failed to set feature flag", "error", err)
		return http.StatusInternalServerError
	}
}

func writeFlagsError(w http.ResponseWriter, status int, msg string) {
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
