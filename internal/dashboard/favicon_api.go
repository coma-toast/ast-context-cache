package dashboard

import (
	"io/fs"
	"net/http"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/instance"
)

// FaviconPath serves the instance favicon; FaviconURL adds the ?v= cache buster.
const FaviconPath = "/dashboard/favicon"

// settingsMaxBodyBytes caps a settings POST: a 64 KB favicon upload is ~88 KB as a data URL.
const settingsMaxBodyBytes = 1 << 20

// isFaviconSettingKey reports whether key is one of the favicon settings.
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

// validateFaviconSetting checks a favicon key against the batch view: format (400), and the
// env lock (409 unless the override is on in the stored settings or the same batch). An upload
// comes back as a normalized data URL.
func validateFaviconSetting(key, value string, view settingsView) (string, error) {
	value = strings.TrimSpace(value)
	locked := instance.FaviconLocked() && view.get(instance.FaviconOverrideKey, "") != "true"
	var err error
	switch {
	case key == instance.FaviconOverrideKey:
		if !validBoolSetting(value) {
			return "", errs.NewCode(errs.CodeInvalidInput, key+" must be true or false")
		}
	case locked:
		return "", errs.NewCode(errs.CodeConflict, key+" is set by "+instance.FaviconColorEnv+" or "+instance.FaviconBadgeEnv+"; turn on "+instance.FaviconOverrideKey+" or clear them first")
	case key == instance.FaviconModeKey:
		err = instance.ValidateFaviconMode(value)
		if err == nil && value == instance.FaviconUpload && view.get(instance.FaviconUploadKey, "") == "" {
			return "", errs.NewCode(errs.CodeInvalidInput, "upload a favicon before choosing upload mode")
		}
	case key == instance.FaviconColorKey:
		value, err = instance.NormalizeFaviconColor(value)
	case key == instance.FaviconBadgeKey:
		value, err = instance.NormalizeFaviconBadge(value)
	case key == instance.FaviconUploadKey && value != "":
		value, _, _, err = instance.ParseFaviconUpload(value)
	}
	if err != nil {
		return "", errs.WrapCode(errs.CodeInvalidInput, err)
	}
	return value, nil
}

// applyFaviconSetting stores a validated favicon key. An upload switches the mode to upload;
// clearing it while in upload mode goes back to the default icon. Open tabs pick up the new
// favicon URL via SettingsChanged.
func applyFaviconSetting(key, value string, extra map[string]string) error {
	mode := ""
	switch {
	case key == instance.FaviconUploadKey && value != "":
		_, ct, body, err := instance.ParseFaviconUpload(value)
		if err != nil {
			return err
		}
		mode, extra["upload_type"], extra["upload_bytes"] = instance.FaviconUpload, ct, strconv.Itoa(len(body))
	case key == instance.FaviconUploadKey && db.GetSetting(instance.FaviconModeKey, "") == instance.FaviconUpload:
		mode = instance.FaviconDefault
	}
	if err := db.SetSetting(key, value); err != nil {
		return err
	}
	if mode == "" {
		return nil
	}
	extra[instance.FaviconModeKey] = mode
	return db.SetSetting(instance.FaviconModeKey, mode)
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
