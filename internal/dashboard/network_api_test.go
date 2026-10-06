package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/netlisten"
)

// No t.Parallel: the db pools, the netlisten registry and httpguard's token and trusted
// set are package globals.

const (
	netTestToken   = "tailnet-token-123"
	netTailnetIP   = "100.105.22.11"
	netTailnetAddr = "100.64.0.9:5555"
	netLoopback    = "127.0.0.1:5555"
	// netUnboundIP is in TEST-NET-2 (RFC 5737), which no host has configured.
	netUnboundIP = "198.51.100.1"
)

type networkResponse struct {
	netlisten.State
	Error string `json:"error"`
}

// setupNetworkAPI clears the network env vars (one left set in the developer's shell
// would lock a key), opens a fresh database, and resets httpguard afterwards.
func setupNetworkAPI(t *testing.T) http.Handler {
	t.Helper()
	for _, key := range netlisten.Keys {
		t.Setenv(netlisten.EnvFor(key), "")
	}
	t.Cleanup(func() {
		httpguard.SetAccessToken("")
		httpguard.SetTrusted(nil)
	})
	dbtest.Init(t)
	return NewHandler("127.0.0.1")
}

// netRequest builds an Origin-less request (as curl sends) from remote.
func netRequest(method, path, body, remote string) *http.Request {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "127.0.0.1:7830"
	req.RemoteAddr = remote
	req.Header.Set("Content-Type", "application/json")
	return req
}

func serveNetwork(t *testing.T, h http.Handler, method, body string) (int, networkResponse, string) {
	t.Helper()
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, netRequest(method, "/api/dashboard/network", body, netLoopback))
	var out networkResponse
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &out), rr.Body.String())
	return rr.Code, out, rr.Body.String()
}

func TestNetworkAPIGetDefaults(t *testing.T) {
	h := setupNetworkAPI(t)
	code, out, _ := serveNetwork(t, h, http.MethodGet, "")
	require.Equal(t, http.StatusOK, code)
	assert.Empty(t, out.ExtraAddrs)
	assert.Empty(t, out.TrustedHosts)
	assert.False(t, out.TokenSet)
	assert.Empty(t, out.Locked)
	assert.NotNil(t, out.Listeners)
}

func TestNetworkAPIPostAppliesAndNeverEchoesToken(t *testing.T) {
	h := setupNetworkAPI(t)
	m := netlisten.NewManager("mcp", &http.Server{Handler: http.NotFoundHandler()}, 0)
	netlisten.Register(m)
	t.Cleanup(func() {
		netlisten.Unregister(m)
		m.Close()
	})
	body := `{"listen_extra_addrs":"` + netUnboundIP + `","trusted_hosts":"Mac.ts.net.","remote_access_token":"` + netTestToken + `"}`
	code, out, raw := serveNetwork(t, h, http.MethodPost, body)
	require.Equal(t, http.StatusOK, code, out.Error)
	assert.NotContains(t, raw, netTestToken, "POST response never echoes the token")
	assert.Equal(t, []string{netUnboundIP}, out.ExtraAddrs)
	assert.Equal(t, []string{"mac.ts.net"}, out.TrustedHosts)
	assert.True(t, out.TokenSet)
	require.Len(t, out.Listeners, 1)
	assert.Equal(t, netlisten.Status{Server: "mcp", Addr: netUnboundIP, Port: 0, Status: netlisten.StatusError, Error: out.Listeners[0].Error}, out.Listeners[0])
	assert.Contains(t, out.Listeners[0].Error, "failed to open extra listener")
	assert.True(t, httpguard.TokenRequired())
	assert.True(t, httpguard.IsTrustedHost("mac.ts.net:7830"))

	code, out, raw = serveNetwork(t, h, http.MethodGet, "")
	require.Equal(t, http.StatusOK, code)
	assert.NotContains(t, raw, netTestToken, "GET never returns the token")
	assert.True(t, out.TokenSet)

	code, out, _ = serveNetwork(t, h, http.MethodPost, `{"remote_access_token":""}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	assert.False(t, out.TokenSet, "empty token clears it")
	assert.Equal(t, []string{netUnboundIP}, out.ExtraAddrs, "keys not sent are unchanged")
	assert.False(t, httpguard.TokenRequired())
}

func TestNetworkAPIRejectsBadValues(t *testing.T) {
	h := setupNetworkAPI(t)
	tests := []struct {
		name string
		body string
		want string
	}{
		{"hostname as address", `{"listen_extra_addrs":"mac.ts.net"}`, "is not an IP address"},
		{"wildcard address", `{"listen_extra_addrs":"0.0.0.0"}`, "is a wildcard"},
		{"loopback address", `{"listen_extra_addrs":"127.0.0.1"}`, "is loopback"},
		{"bad hostname", `{"trusted_hosts":"bad_host.ts.net"}`, "is not a valid hostname"},
		{"token with space", `{"remote_access_token":"a b"}`, "whitespace"},
		{"nothing", `{}`, "no network setting given"},
		{"bad json", `{`, "invalid JSON body"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, out, _ := serveNetwork(t, h, http.MethodPost, tt.body)
			require.Equal(t, http.StatusBadRequest, code)
			assert.Contains(t, out.Error, tt.want)
		})
	}
	assert.Empty(t, db.GetSetting(netlisten.KeyExtraAddrs, ""))
}

func TestNetworkAPIEnvLocked(t *testing.T) {
	h := setupNetworkAPI(t)
	t.Setenv(netlisten.EnvExtraAddrs, netUnboundIP)
	t.Setenv(netlisten.EnvAccessToken, netTestToken)
	code, out, _ := serveNetwork(t, h, http.MethodPost, `{"listen_extra_addrs":"100.64.0.1"}`)
	require.Equal(t, http.StatusConflict, code)
	assert.Contains(t, out.Error, netlisten.EnvExtraAddrs)
	code, out, raw := serveNetwork(t, h, http.MethodGet, "")
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, []string{netUnboundIP}, out.ExtraAddrs)
	assert.True(t, out.TokenSet)
	assert.NotContains(t, raw, netTestToken)
	assert.Equal(t, map[string]string{
		netlisten.KeyExtraAddrs:  netlisten.EnvExtraAddrs,
		netlisten.KeyAccessToken: netlisten.EnvAccessToken,
	}, out.Locked)

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, netRequest(http.MethodPost, "/api/settings", `{"key":"remote_access_token","value":"x"}`, netLoopback))
	assert.Equal(t, http.StatusConflict, rr.Code, "the generic settings POST honors the lock too")
}

func TestSettingsAPIHidesToken(t *testing.T) {
	h := setupNetworkAPI(t)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, netRequest(http.MethodPost, "/api/settings", `{"key":"remote_access_token","value":"`+netTestToken+`"}`, netLoopback))
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	assert.NotContains(t, rr.Body.String(), netTestToken, "settings POST doesn't echo the token")
	assert.Equal(t, netTestToken, db.GetSetting(netlisten.KeyAccessToken, ""))
	assert.True(t, httpguard.TokenRequired(), "generic settings POST applies live")

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, netRequest(http.MethodGet, "/api/settings", "", netLoopback))
	require.Equal(t, http.StatusOK, rr.Code)
	assert.NotContains(t, rr.Body.String(), netTestToken)
	var settings map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &settings))
	_, has := settings[netlisten.KeyAccessToken]
	assert.False(t, has)
	assert.Equal(t, "true", settings[tokenSetSettingKey])

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, netRequest(http.MethodPost, "/api/settings", `{"key":"listen_extra_addrs","value":"not-an-ip"}`, netLoopback))
	assert.Equal(t, http.StatusBadRequest, rr.Code, "generic settings POST validates network keys")
}

func TestRemoteLoginFlow(t *testing.T) {
	h := setupNetworkAPI(t)
	require.NoError(t, netlisten.Save(map[string]string{netlisten.KeyExtraAddrs: netTailnetIP, netlisten.KeyAccessToken: netTestToken}))
	remote := func(method, path, body string) *http.Request {
		req := netRequest(method, path, body, netTailnetAddr)
		req.Host = netTailnetIP + ":7830"
		return req
	}

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, remote(http.MethodGet, "/dashboard/", ""))
	require.Equal(t, http.StatusSeeOther, rr.Code)
	assert.Equal(t, httpguard.LoginPath, rr.Header().Get("Location"))

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, remote(http.MethodGet, "/api/dashboard/network", ""))
	require.Equal(t, http.StatusUnauthorized, rr.Code)

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, remote(http.MethodGet, httpguard.LoginPath, ""))
	require.Equal(t, http.StatusOK, rr.Code)
	assert.Contains(t, rr.Body.String(), "<form")

	login := remote(http.MethodPost, httpguard.LoginPath, url.Values{"token": {netTestToken}}.Encode())
	login.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	login.Header.Set("Origin", "http://"+netTailnetIP+":7830")
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, login)
	require.Equal(t, http.StatusSeeOther, rr.Code, rr.Body.String())
	cookies := rr.Result().Cookies()
	require.Len(t, cookies, 1)
	assert.NotContains(t, cookies[0].Value, netTestToken)

	authed := remote(http.MethodGet, "/api/dashboard/network", "")
	authed.AddCookie(cookies[0])
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, authed)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	assert.NotContains(t, rr.Body.String(), netTestToken)

	bearer := remote(http.MethodGet, "/api/dashboard/network", "")
	bearer.Header.Set("Authorization", "Bearer "+netTestToken)
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, bearer)
	assert.Equal(t, http.StatusOK, rr.Code)

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, netRequest(http.MethodGet, "/api/dashboard/network", "", netLoopback))
	assert.Equal(t, http.StatusOK, rr.Code, "loopback never needs the token")
}
