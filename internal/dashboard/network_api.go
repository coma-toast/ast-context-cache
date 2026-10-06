package dashboard

import (
	"encoding/json"
	"net/http"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/netlisten"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// tokenSetSettingKey replaces the access token in the generic settings GET, which must
// never return the token itself.
const tokenSetSettingKey = netlisten.KeyAccessToken + "_set"

// networkRequest is a partial update: only the keys present are changed. Lists are
// comma or newline separated; an empty token disables token auth.
type networkRequest struct {
	ExtraAddrs   *string `json:"listen_extra_addrs"`
	TrustedHosts *string `json:"trusted_hosts"`
	AccessToken  *string `json:"remote_access_token"`
}

func (n networkRequest) values() map[string]string {
	out := map[string]string{}
	for key, v := range map[string]*string{
		netlisten.KeyExtraAddrs:   n.ExtraAddrs,
		netlisten.KeyTrustedHosts: n.TrustedHosts,
		netlisten.KeyAccessToken:  n.AccessToken,
	} {
		if v != nil {
			out[key] = *v
		}
	}
	return out
}

// handleDashboardNetworkJSON reports the network access settings and live extra
// listeners (GET) or updates some of them (POST), applying the change immediately. An
// invalid value answers 400 and an env-locked key 409. The token is never returned.
func handleDashboardNetworkJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		json.NewEncoder(w).Encode(netlisten.CurrentState())
	case http.MethodPost:
		var req networkRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
			writeFlagsError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		if err := netlisten.Save(req.values()); err != nil {
			writeFlagsError(w, networkErrorStatus(err), err.Error())
			return
		}
		realtime.Notify(realtime.SettingsChanged)
		json.NewEncoder(w).Encode(netlisten.CurrentState())
	default:
		writeFlagsError(w, http.StatusMethodNotAllowed, "GET or POST required")
	}
}

// saveNetworkSetting is the generic settings POST's path for a network key, so both
// endpoints validate, honor env locks and apply live the same way.
func saveNetworkSetting(w http.ResponseWriter, key, value string) {
	if err := netlisten.Save(map[string]string{key: value}); err != nil {
		writeFlagsError(w, networkErrorStatus(err), err.Error())
		return
	}
	shown, _ := netlisten.Normalize(key, value)
	if key == netlisten.KeyAccessToken {
		shown = ""
	}
	writeSettingsOK(w, map[string]string{"key": key, "value": shown}, false, realtime.SettingsChanged)
}

// redactNetworkSettings swaps the stored access token for a set/unset marker.
func redactNetworkSettings(settings map[string]string) {
	tokenSet := netlisten.Load().Token != ""
	delete(settings, netlisten.KeyAccessToken)
	settings[tokenSetSettingKey] = "false"
	if tokenSet {
		settings[tokenSetSettingKey] = "true"
	}
}

func networkErrorStatus(err error) int {
	switch {
	case errs.HasCode(err, errs.CodeConflict):
		return http.StatusConflict
	case errs.HasCode(err, errs.CodeInvalidInput), errs.HasCode(err, errs.CodeNotFound):
		return http.StatusBadRequest
	default:
		logger.Warn("Failed to save network settings", "error", err)
		return http.StatusInternalServerError
	}
}
