package dashboard

import (
	"net/http"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// saveInstanceName handles the generic settings POST for the instance_name key:
// validation, the env/flag lock (409), and live application. Renaming re-issues
// the session cookie under the new scoped name and expires the old one, so the
// user stays logged in. Open tabs learn the new name via SettingsChanged.
func saveInstanceName(w http.ResponseWriter, r *http.Request, value string) {
	value = strings.TrimSpace(value)
	if instance.IsLocked() {
		writeFlagsError(w, http.StatusConflict, "instance_name is set by the -name flag or AST_INSTANCE_NAME; clear that first")
		return
	}
	if value != "" {
		if ok, err := instance.ValidateName(value); !ok {
			writeFlagsError(w, http.StatusBadRequest, "invalid instance_name: "+err.Error())
			return
		}
	}
	if err := db.SetSetting("instance_name", value); err != nil {
		writeFlagsError(w, http.StatusInternalServerError, err.Error())
		return
	}
	n := instance.Update(value, db.GetDataDir())
	httpguard.SetInstanceName(n.Value)
	httpguard.RotateCookieScope(w, r, n.Scope)
	writeSettingsOK(w, map[string]string{
		"key":                     "instance_name",
		"value":                   value,
		"instance_name_effective": n.Value,
		"instance_scope":          n.Scope,
	}, false, realtime.SettingsChanged)
}

// applyInstanceSettings enriches the generic settings GET with the effective
// instance name and its lock state, so the UI can render the field read-only
// with a "set by flag/env" hint, the same way network settings handle locks.
func applyInstanceSettings(settings map[string]string) {
	cur := instance.Current()
	if cur.Value == "" && cur.Scope == "" {
		// Update was never called (e.g. in tests): resolve from the DB setting.
		cur = instance.Update(strings.TrimSpace(db.GetSetting("instance_name", "")), db.GetDataDir())
	}
	if _, ok := settings["instance_name"]; !ok || settings["instance_name"] == "" {
		settings["instance_name"] = strings.TrimSpace(db.GetSetting("instance_name", ""))
	}
	settings["instance_name_effective"] = cur.Value
	settings["instance_scope"] = cur.Scope
	settings["instance_name_source"] = cur.Source
	settings["instance_name_locked"] = "false"
	if instance.IsLocked() {
		settings["instance_name_locked"] = "true"
	}
}
