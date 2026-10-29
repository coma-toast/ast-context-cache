package dashboard

import (
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/netlisten"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/transcripts"
)

// secretSettingKeys hold credentials: their values are never logged or echoed back.
var secretSettingKeys = map[string]bool{"EMBED_HTTP_BEARER": true, "EMBED_OPENAI_API_KEY": true, netlisten.KeyAccessToken: true}

// boolSettingKeys must be "true" or "false".
var boolSettingKeys = map[string]bool{transcripts.SettingKey: true, "index_log_files": true, "log_retention_enabled": true, "log_retention_dry_run": true, "query_retention_enabled": true}

// countSettingKeys are plain integer settings where 0 means off or unlimited.
var countSettingKeys = map[string]bool{
	"idle_unload_minutes": true, "context_max_notes_session": true, "context_max_tokens_session": true, "context_max_notes_global": true,
	"context_max_tokens_global": true, "query_retention_max_age_days": true, "log_retention_max_age_days": true, "log_retention_max_total_mib": true,
	"dashboard_log_tail_lines": true, "dashboard_log_line_chars": true,
}

// listSettingKeys hold a JSON array of strings; empty means the default.
var listSettingKeys = map[string]bool{"watcher_ignore_globs": true, "project_exclude_paths": true, "log_retention_roots": true}

// embedBackends are the EMBED_BACKEND / EMBED_AUX_BACKEND values the embedder can build.
var embedBackends = []string{"onnx", "http", "ollama", "openai", "docker"}

// reloadEmbedder rebuilds the embedder from the stored settings; tests swap it to count reloads.
var reloadEmbedder = embedder.Reload

// settingsView reads a setting as it will be once the batch being validated is applied.
type settingsView map[string]string

func (v settingsView) get(key, def string) string {
	if val, ok := v[key]; ok {
		return strings.TrimSpace(val)
	}
	return db.GetSetting(key, def)
}

// settingsApplied is what applying a batch changed: the values to echo back (secrets blank),
// extra response fields, the realtime panels to refresh, and whether the embedder must reload.
type settingsApplied struct {
	values map[string]string
	extra  map[string]string
	notify realtime.Reason
	reload bool
}

// handleSettingsBatch saves several settings in one request: {"settings": {key: value}}. Every
// key is validated against the batch before any is written; one bad value rejects the whole
// batch with 400 (409 when only env locks refused it) and per-key messages in "errors".
func handleSettingsBatch(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		writeFlagsError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req struct {
		Settings map[string]string `json:"settings"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, settingsMaxBodyBytes)).Decode(&req); err != nil {
		logger.Warn("Rejected settings save", "reason", "invalid request body", "error", err)
		writeFlagsError(w, http.StatusBadRequest, "invalid settings request: "+err.Error())
		return
	}
	if len(req.Settings) == 0 {
		logger.Warn("Rejected settings save", "reason", "no settings given")
		writeFlagsError(w, http.StatusBadRequest, "settings required")
		return
	}
	saveSettings(w, r, req.Settings, false)
}

// saveSettings validates every value, then applies them all and answers with what was saved.
// single keeps the one-key response shape ("key", "value") of POST /api/settings.
func saveSettings(w http.ResponseWriter, r *http.Request, in map[string]string, single bool) {
	vals, rejected := validateSettings(in)
	if len(rejected) > 0 {
		writeSettingsRejected(w, in, rejected, single)
		return
	}
	res, key, err := applySettings(w, r, vals)
	if err != nil {
		msg := redactSecret(key, in[key], err.Error())
		status := settingErrorStatus(err, http.StatusInternalServerError)
		logger.Warn("Failed to apply settings", "key", key, "status", status, "error", msg)
		writeSettingsErrors(w, status, msg, map[string]string{key: msg}, single)
		return
	}
	out := map[string]interface{}{"status": "ok"}
	for k, v := range res.extra {
		out[k] = v
	}
	if single {
		for k, v := range res.values {
			out["key"], out["value"] = k, v
		}
	} else {
		out["values"] = res.values
	}
	if res.reload {
		appendReloadFields(out)
		res.notify |= realtime.IndexHealth | realtime.HealthBar
	}
	realtime.Notify(res.notify)
	json.NewEncoder(w).Encode(out)
}

// validateSettings normalizes every value or returns per-key errors; it writes nothing.
func validateSettings(in map[string]string) (map[string]string, map[string]error) {
	view := settingsView(in)
	vals, rejected := make(map[string]string, len(in)), map[string]error{}
	for key, value := range in {
		v, err := validateSetting(key, value, view)
		if err != nil {
			rejected[key] = err
			continue
		}
		vals[key] = v
	}
	return vals, rejected
}

// validateSetting checks one value against the batch view and returns the form to store.
// Errors carry CodeInvalidInput (400) or CodeConflict (409, an env var or flag owns the key).
func validateSetting(key, value string, view settingsView) (string, error) {
	switch {
	case key == "":
		return "", errs.NewCode(errs.CodeInvalidInput, "key required")
	case isFlagKey(key):
		return "", errs.NewCode(errs.CodeInvalidInput, flagSettingKeyMsg)
	case netlisten.IsKey(key):
		return validateNetworkSetting(key, value)
	case embedqueue.IsQueueCapSetting(key):
		return validateQueueCapSetting(key, value)
	case isInstanceSettingKey(key):
		return validateInstanceSetting(key, value, view)
	case isFaviconSettingKey(key):
		return validateFaviconSetting(key, value, view)
	case handoff.IsLimitSetting(key):
		return parseIntSetting(value, 1, 1<<31-1, key+" must be a positive integer")
	case boolSettingKeys[key]:
		if v := strings.TrimSpace(value); validBoolSetting(v) {
			return v, nil
		}
		return "", errs.NewCode(errs.CodeInvalidInput, key+" must be true or false")
	case countSettingKeys[key]:
		return parseIntSetting(value, 0, 1<<31-1, key+" must be a whole number, 0 or more")
	case listSettingKeys[key]:
		return validateListSetting(key, value)
	case key == "embed_worker_max":
		return parseIntSetting(value, 1, embedqueue.AbsoluteMaxWorkers, fmt.Sprintf("embed_worker_max must be 1–%d", embedqueue.AbsoluteMaxWorkers))
	case key == "embed_aux_worker_max":
		return parseIntSetting(value, 1, 32, "embed_aux_worker_max must be 1–32")
	case key == "EMBED_AUX_WORKERS":
		max := embedqueue.AuxMaxWorkers()
		if raw, ok := view["embed_aux_worker_max"]; ok {
			if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil {
				max = n
			}
		}
		return parseIntSetting(value, embedqueue.MinWorkers, max, fmt.Sprintf("EMBED_AUX_WORKERS must be %d–%d", embedqueue.MinWorkers, max))
	case key == embedder.ProbeIntervalSettingKey:
		return parseIntSetting(value, 5, 600, key+" must be 5–600")
	case key == "EMBED_BACKEND" || key == "EMBED_AUX_BACKEND":
		v := normalizeEmbedBackendValue(value)
		if v != "" && !slices.Contains(embedBackends, components.EmbedBackendUI(v)) {
			return "", errs.NewCode(errs.CodeInvalidInput, key+" must be one of "+strings.Join(embedBackends, ", "))
		}
		return v, nil
	}
	return value, nil
}

// parseIntSetting parses value as an integer in [lo, hi], failing with msg.
func parseIntSetting(value string, lo, hi int, msg string) (string, error) {
	n, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || n < lo || n > hi {
		return "", errs.NewCode(errs.CodeInvalidInput, msg)
	}
	return strconv.Itoa(n), nil
}

// validateListSetting accepts empty or a JSON array of strings.
func validateListSetting(key, value string) (string, error) {
	v := strings.TrimSpace(value)
	if v == "" {
		return v, nil
	}
	var list []string
	if err := json.Unmarshal([]byte(v), &list); err != nil {
		return "", errs.NewCode(errs.CodeInvalidInput, key+` must be a JSON array of strings, e.g. ["**/node_modules/**"]`)
	}
	return v, nil
}

// validateNetworkSetting normalizes a network access key; an env var set for it locks it (409).
func validateNetworkSetting(key, value string) (string, error) {
	if env := netlisten.EnvFor(key); env != "" && strings.TrimSpace(os.Getenv(env)) != "" {
		return "", errs.NewCode(errs.CodeConflict, key+" is locked by "+env)
	}
	return netlisten.Normalize(key, value)
}

// validateQueueCapSetting checks an embed queue soft limit: 1..hard max, and not env locked (409).
func validateQueueCapSetting(key, value string) (string, error) {
	n, err := embedqueue.ValidateQueueCap(key, value)
	if err != nil {
		return "", err
	}
	for _, c := range embedqueue.QueueCaps() {
		if c.Key == key && embedqueue.QueueCapEnvLocked(c) {
			return "", errs.NewCode(errs.CodeConflict, fmt.Sprintf("%s is set by %s", key, c.Env))
		}
	}
	return strconv.Itoa(n), nil
}

// settingApplyRank orders a batch so a key is written before the keys that depend on it:
// overrides before the values they unlock, the backend before its fields, a max before its count.
func settingApplyRank(key string) int {
	switch key {
	case instance.OverrideKey, instance.FaviconOverrideKey, instance.FaviconUploadKey, "EMBED_BACKEND", "embed_worker_max", "embed_aux_worker_max":
		return 0
	}
	return 1
}

// applySettings writes validated values and runs each key's side effects once per batch: one
// embedder reload, one network apply, one instance-name refresh. It returns the failing key.
func applySettings(w http.ResponseWriter, r *http.Request, vals map[string]string) (settingsApplied, string, error) {
	out := settingsApplied{values: map[string]string{}, extra: map[string]string{}, notify: realtime.SettingsChanged}
	keys := slices.Sorted(maps.Keys(vals))
	slices.SortStableFunc(keys, func(a, b string) int { return settingApplyRank(a) - settingApplyRank(b) })
	net := map[string]string{}
	for _, k := range keys {
		if netlisten.IsKey(k) {
			net[k] = vals[k]
		}
	}
	if len(net) > 0 {
		if err := netlisten.Save(net); err != nil {
			return out, slices.Sorted(maps.Keys(net))[0], err
		}
	}
	var instanceChanged, faviconChanged, embedChanged bool
	for _, key := range keys {
		value := vals[key]
		out.values[key] = value
		if secretSettingKeys[key] || key == instance.FaviconUploadKey {
			out.values[key] = ""
		}
		var err error
		switch {
		case netlisten.IsKey(key):
		case embedqueue.IsQueueCapSetting(key):
			_, err = embedqueue.SetQueueCap(key, value)
			out.notify |= realtime.IndexHealth | realtime.HealthBar
		case isInstanceSettingKey(key):
			err, instanceChanged = db.SetSetting(key, value), true
		case isFaviconSettingKey(key):
			err, faviconChanged = applyFaviconSetting(key, value, out.extra), true
		case key == "EMBED_BACKEND":
			old := components.EmbedBackendUI(db.GetSetting("EMBED_BACKEND", ""))
			err, out.reload = switchEmbedBackend(old, components.EmbedBackendUI(value), value), true
		default:
			if err = db.SetSetting(key, value); err == nil {
				err = applySettingSideEffects(key, value)
			}
			embedChanged = embedChanged || embedder.IsSettingKey(key)
			out.notify |= settingNotifyMask(key)
		}
		if err != nil {
			return out, key, err
		}
	}
	if embedChanged {
		snapshotEmbedProfileFromDB(db.GetSetting("EMBED_BACKEND", ""))
		out.reload = true
	}
	if instanceChanged {
		n := updateInstanceName()
		httpguard.SetInstanceName(n.Value)
		httpguard.RotateCookieScope(w, r, n.Scope)
		out.extra["instance_name_effective"], out.extra["instance_name_source"], out.extra["instance_scope"] = n.Value, n.Source, n.Scope
	}
	if faviconChanged {
		f := currentFavicon()
		logger.Info("Saved favicon settings", "mode", f.Mode, "source", f.Source, "version", f.Version())
		out.extra["favicon_mode"], out.extra["favicon_source"], out.extra["favicon_url"] = f.Mode, f.Source, faviconURL(f)
	}
	return out, "", nil
}

// applySettingSideEffects refreshes the caches and live workers a stored key feeds.
func applySettingSideEffects(key, value string) error {
	switch key {
	case "watcher_ignore_globs":
		ignorepatterns.InvalidateCache()
		indexer.InvalidateAllPathFilters()
	case "project_exclude_paths":
		projectmeta.InvalidateExcludeCache()
	case "embed_worker_max":
		return embedqueue.ClampWorkersToMax()
	case "embed_aux_worker_max":
		return embedqueue.ClampAuxWorkersToMax()
	case "EMBED_AUX_WORKERS":
		n, _ := strconv.Atoi(value)
		if _, err := embedqueue.SetAuxWorkerCount(n); err != nil {
			logger.Warn("Failed to set aux embed workers", "workers", n, "error", err)
		}
	case "EMBED_AUX_BACKEND":
		if !embedder.AuxRuntimeReady() {
			return nil
		}
		if err := embedder.ReloadAuxRuntime(); err != nil {
			return err
		}
		embedqueue.SetAuxEmbedder(embedder.RawAux())
	case embedder.ProbeIntervalSettingKey:
		embedder.RestartConnectivityProbe()
	}
	return nil
}

// settingNotifyMask is the realtime panels a stored key changes, beyond SettingsChanged.
func settingNotifyMask(key string) realtime.Reason {
	switch {
	case embedder.IsSettingKey(key), key == "idle_unload_minutes", key == "embed_worker_max", key == "embed_aux_worker_max",
		key == "EMBED_AUX_WORKERS", key == "EMBED_AUX_BACKEND", key == embedder.ProbeIntervalSettingKey:
		return realtime.IndexHealth | realtime.HealthBar
	case key == "project_exclude_paths":
		return realtime.IndexHealth
	case handoff.IsLimitSetting(key):
		return realtime.Handoffs
	}
	return 0
}

// settingErrorStatus maps a settings error to its HTTP status, def when it has no client code.
func settingErrorStatus(err error, def int) int {
	switch {
	case errs.HasCode(err, errs.CodeConflict):
		return http.StatusConflict
	case errs.HasCode(err, errs.CodeInvalidInput), errs.HasCode(err, errs.CodeNotFound):
		return http.StatusBadRequest
	}
	return def
}

// writeSettingsRejected logs each rejected key at WARN (never a secret value) and answers 400,
// or 409 when env locks were the only reason.
func writeSettingsRejected(w http.ResponseWriter, in map[string]string, rejected map[string]error, single bool) {
	status, fields, msgs := http.StatusConflict, map[string]string{}, []string{}
	for _, key := range slices.Sorted(maps.Keys(rejected)) {
		code := settingErrorStatus(rejected[key], http.StatusBadRequest)
		if code != http.StatusConflict {
			status = http.StatusBadRequest
		}
		msg := redactSecret(key, in[key], rejected[key].Error())
		logger.Warn("Rejected settings save", "key", key, "status", code, "reason", msg)
		fields[key] = msg
		msgs = append(msgs, msg)
	}
	writeSettingsErrors(w, status, strings.Join(msgs, "; "), fields, single)
}

// writeSettingsErrors answers {"error", "field": first key}, plus "errors": {key: message} for
// a batch (the one-key POST keeps a flat string map for its existing callers).
func writeSettingsErrors(w http.ResponseWriter, status int, msg string, fields map[string]string, single bool) {
	w.WriteHeader(status)
	out := map[string]interface{}{"error": msg}
	if keys := slices.Sorted(maps.Keys(fields)); len(keys) > 0 {
		out["field"] = keys[0]
	}
	if !single {
		out["errors"] = fields
	}
	json.NewEncoder(w).Encode(out)
}

// redactSecret strips a secret key's value from a message before it is logged or returned.
func redactSecret(key, value, msg string) string {
	if !secretSettingKeys[key] || strings.TrimSpace(value) == "" {
		return msg
	}
	return strings.ReplaceAll(msg, strings.TrimSpace(value), "[redacted]")
}
