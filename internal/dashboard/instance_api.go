package dashboard

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// StoredInstanceName returns the instance_name setting and whether the
// instance_name_override setting is on, in the order instance.Update takes them.
func StoredInstanceName() (string, bool) {
	return strings.TrimSpace(db.GetSetting(instance.SettingKey, "")), db.GetSetting(instance.OverrideKey, "") == "true"
}

// isInstanceSettingKey reports whether the settings POST for key goes to saveInstanceSetting.
func isInstanceSettingKey(key string) bool {
	return key == instance.SettingKey || key == instance.OverrideKey
}

// updateInstanceName re-resolves the live instance name from the stored settings.
func updateInstanceName() instance.Name {
	name, override := StoredInstanceName()
	return instance.Update(name, override, db.GetDataDir())
}

// instanceNameWriteLocked is true when -name or AST_INSTANCE_NAME sets the name
// and the override is off, so the dashboard may not change instance_name.
func instanceNameWriteLocked() bool {
	_, override := StoredInstanceName()
	return instance.IsLocked() && !override
}

// saveInstanceSetting handles the generic settings POST for instance_name and
// instance_name_override: validation, the env/flag lock (409 unless the override
// is on), and live application. A changed name re-issues the session cookie under
// the new scoped name and expires the old one, so the user stays logged in. Open
// tabs learn the new name via SettingsChanged.
func saveInstanceSetting(w http.ResponseWriter, r *http.Request, key, value string) {
	value = strings.TrimSpace(value)
	switch {
	case key == instance.OverrideKey && !validBoolSetting(value):
		writeFlagsError(w, http.StatusBadRequest, key+" must be true or false")
		return
	case key == instance.SettingKey && instanceNameWriteLocked():
		writeFlagsError(w, http.StatusConflict, "instance_name is set by the -name flag or AST_INSTANCE_NAME; turn on instance_name_override or clear that first")
		return
	case key == instance.SettingKey && value != "":
		if ok, err := instance.ValidateName(value); !ok {
			writeFlagsError(w, http.StatusBadRequest, "invalid instance_name: "+err.Error())
			return
		}
	}
	if err := db.SetSetting(key, value); err != nil {
		writeFlagsError(w, http.StatusInternalServerError, err.Error())
		return
	}
	n := updateInstanceName()
	httpguard.SetInstanceName(n.Value)
	httpguard.RotateCookieScope(w, r, n.Scope)
	writeSettingsOK(w, map[string]string{
		"key":                     key,
		"value":                   value,
		"instance_name_effective": n.Value,
		"instance_name_source":    n.Source,
		"instance_scope":          n.Scope,
	}, false, realtime.SettingsChanged)
}

// currentInstanceName returns the live name, resolving it from the settings when
// Update was never called (e.g. in tests).
func currentInstanceName() instance.Name {
	if cur := instance.Current(); cur.Value != "" || cur.Scope != "" {
		return cur
	}
	return updateInstanceName()
}

// populateInstanceData fills the instance name fields of the settings payload.
func populateInstanceData(data *components.SettingsData) {
	cur := currentInstanceName()
	data.InstanceName, data.InstanceNameOverride = StoredInstanceName()
	data.InstanceNameEffective, data.InstanceScope, data.InstanceNameSource = cur.Value, cur.Scope, cur.Source
	data.InstanceNameExternal, data.InstanceNameExternalSource = instance.External()
	data.InstanceNameLocked = instanceNameWriteLocked()
	populateFaviconData(data)
}

// applyInstanceSettings enriches the generic settings GET with the effective
// instance name, its source, the override and the write lock, so the UI can
// render the field read-only with a "set by flag/env" hint and the override box.
func applyInstanceSettings(settings map[string]string) {
	var data components.SettingsData
	populateInstanceData(&data)
	settings[instance.SettingKey] = data.InstanceName
	settings[instance.OverrideKey] = strconv.FormatBool(data.InstanceNameOverride)
	settings["instance_name_effective"] = data.InstanceNameEffective
	settings["instance_scope"] = data.InstanceScope
	settings["instance_name_source"] = data.InstanceNameSource
	settings["instance_name_external"] = data.InstanceNameExternal
	settings["instance_name_locked"] = strconv.FormatBool(data.InstanceNameLocked)
}
