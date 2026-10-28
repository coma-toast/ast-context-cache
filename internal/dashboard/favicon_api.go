package dashboard

import (
	"io/fs"
	"net/http"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// FaviconPath serves the instance favicon; FaviconURL adds the ?v= cache buster.
const FaviconPath = "/dashboard/favicon"

// settingsMaxBodyBytes caps a settings POST: a 64 KB favicon upload is ~88 KB as a data URL.
const settingsMaxBodyBytes = 1 << 20

// isFaviconSettingKey reports whether the settings POST for key goes to saveFaviconSetting.
func isFaviconSettingKey(key string) bool {
	switch key {
	case instance.FaviconModeKey, instance.FaviconColorKey, instance.FaviconBadgeKey, instance.FaviconUploadKey, instance.FaviconOverrideKey:
		return true
	}
	return false
}

// storedFavicon reads the favicon settings.
func storedFavicon() instance.FaviconSettings {
	return instance.FaviconSettings{
		Mode:     db.GetSetting(instance.FaviconModeKey, ""),
		Color:    db.GetSetting(instance.FaviconColorKey, ""),
		Badge:    db.GetSetting(instance.FaviconBadgeKey, ""),
		Upload:   db.GetSetting(instance.FaviconUploadKey, ""),
		Override: db.GetSetting(instance.FaviconOverrideKey, "") == "true",
	}
}

// defaultFaviconSVG is today's icon from the embedded UI build.
func defaultFaviconSVG() []byte {
	if uiFS == nil {
		initUIAssets()
	}
	b, _ := fs.ReadFile(uiFS, "favicon.svg")
	return b
}

// currentFavicon resolves the favicon in effect from the env vars and the settings.
func currentFavicon() instance.Favicon {
	return instance.ResolveFavicon(storedFavicon(), defaultFaviconSVG())
}

// faviconURL is the favicon's stable path with its content hash as the cache buster.
func faviconURL(f instance.Favicon) string {
	return FaviconPath + "?v=" + f.Version()
}

// faviconWriteLocked is true when an AST_INSTANCE_FAVICON_* env var sets the
// favicon and the override is off, so the dashboard may not change it.
func faviconWriteLocked() bool {
	return instance.FaviconLocked() && db.GetSetting(instance.FaviconOverrideKey, "") != "true"
}

// saveFaviconSetting handles the generic settings POST for the favicon keys:
// validation (400), the env lock (409 unless the override is on) and storage.
// An upload is stored as a normalized data URL and switches the mode to upload;
// clearing it while in upload mode goes back to the default icon. Open tabs pick
// up the new favicon URL via SettingsChanged.
func saveFaviconSetting(w http.ResponseWriter, key, value string) {
	value = strings.TrimSpace(value)
	extra := map[string]string{}
	var err error
	switch {
	case key == instance.FaviconOverrideKey:
		if !validBoolSetting(value) {
			writeFlagsError(w, http.StatusBadRequest, key+" must be true or false")
			return
		}
	case faviconWriteLocked():
		writeFlagsError(w, http.StatusConflict, key+" is set by "+instance.FaviconColorEnv+" or "+instance.FaviconBadgeEnv+"; turn on "+instance.FaviconOverrideKey+" or clear them first")
		return
	case key == instance.FaviconModeKey:
		err = instance.ValidateFaviconMode(value)
		if err == nil && value == instance.FaviconUpload && db.GetSetting(instance.FaviconUploadKey, "") == "" {
			writeFlagsError(w, http.StatusBadRequest, "upload a favicon before choosing upload mode")
			return
		}
	case key == instance.FaviconColorKey:
		value, err = instance.NormalizeFaviconColor(value)
	case key == instance.FaviconBadgeKey:
		value, err = instance.NormalizeFaviconBadge(value)
	case key == instance.FaviconUploadKey && value != "":
		var ct string
		var body []byte
		if value, ct, body, err = instance.ParseFaviconUpload(value); err == nil {
			extra[instance.FaviconModeKey], extra["upload_type"], extra["upload_bytes"] = instance.FaviconUpload, ct, strconv.Itoa(len(body))
		}
	case key == instance.FaviconUploadKey && db.GetSetting(instance.FaviconModeKey, "") == instance.FaviconUpload:
		extra[instance.FaviconModeKey] = instance.FaviconDefault
	}
	if err != nil {
		writeFlagsError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := db.SetSetting(key, value); err != nil {
		writeFlagsError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if mode, ok := extra[instance.FaviconModeKey]; ok {
		if err := db.SetSetting(instance.FaviconModeKey, mode); err != nil {
			writeFlagsError(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	f := currentFavicon()
	logger.Info("Saved favicon setting", "key", key, "mode", f.Mode, "source", f.Source, "version", f.Version())
	if key == instance.FaviconUploadKey {
		value = ""
	}
	extra["key"], extra["value"] = key, value
	extra["favicon_mode"], extra["favicon_source"], extra["favicon_url"] = f.Mode, f.Source, faviconURL(f)
	writeSettingsOK(w, extra, false, realtime.SettingsChanged)
}

// handleFavicon serves the favicon in effect. A matching ?v= is cached for a year
// (the URL changes with the content); anything else must revalidate. Uploaded SVG
// is checked on upload, and the CSP sandbox keeps it inert even when opened directly.
func handleFavicon(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	f := currentFavicon()
	w.Header().Set("Content-Type", f.ContentType)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; sandbox")
	w.Header().Set("ETag", `"`+f.Version()+`"`)
	if v := r.URL.Query().Get("v"); v == f.Version() {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		w.Header().Set("Cache-Control", "no-cache")
	}
	if r.Header.Get("If-None-Match") == `"`+f.Version()+`"` {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(f.Body)))
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(f.Body)
}

// populateFaviconData fills the favicon fields of the settings payload.
func populateFaviconData(data *components.SettingsData) {
	s, f := storedFavicon(), currentFavicon()
	data.InstanceFaviconMode, data.InstanceFaviconColor, data.InstanceFaviconBadge = s.Mode, s.Color, s.Badge
	data.InstanceFaviconOverride, data.InstanceFaviconLocked = s.Override, faviconWriteLocked()
	data.InstanceFaviconExternalColor, data.InstanceFaviconExternalBadge = instance.FaviconExternal()
	data.InstanceFaviconEffectiveMode, data.InstanceFaviconSource, data.InstanceFaviconURL = f.Mode, f.Source, faviconURL(f)
	if _, ct, body, err := instance.ParseFaviconUpload(s.Upload); s.Upload != "" && err == nil {
		data.InstanceFaviconUploadType, data.InstanceFaviconUploadBytes = ct, len(body)
	}
}

// applyFaviconSettings adds the favicon state to the generic settings GET. The
// upload's bytes are replaced by its type and size; fetch FaviconPath for them.
func applyFaviconSettings(settings map[string]string) {
	var data components.SettingsData
	populateFaviconData(&data)
	settings[instance.FaviconModeKey] = data.InstanceFaviconMode
	settings[instance.FaviconColorKey] = data.InstanceFaviconColor
	settings[instance.FaviconBadgeKey] = data.InstanceFaviconBadge
	settings[instance.FaviconUploadKey] = ""
	settings[instance.FaviconOverrideKey] = strconv.FormatBool(data.InstanceFaviconOverride)
	settings["instance_favicon_upload_type"] = data.InstanceFaviconUploadType
	settings["instance_favicon_upload_bytes"] = strconv.Itoa(data.InstanceFaviconUploadBytes)
	settings["instance_favicon_effective_mode"] = data.InstanceFaviconEffectiveMode
	settings["instance_favicon_source"] = data.InstanceFaviconSource
	settings["instance_favicon_url"] = data.InstanceFaviconURL
	settings["instance_favicon_locked"] = strconv.FormatBool(data.InstanceFaviconLocked)
}
