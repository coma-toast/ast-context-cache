package netlisten

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
)

// testNet2Addr is in TEST-NET-2 (RFC 5737), which no host has configured.
const testNet2Addr = "198.51.100.1"

// No t.Parallel in the tests below: they share the db pools, the registry and httpguard's
// trusted set and token, all package globals.

// setupConfig clears the env vars (one left set in the developer's shell would lock a key),
// opens a fresh database, and resets the registry and httpguard state afterwards.
func setupConfig(t *testing.T) {
	t.Helper()
	for _, key := range Keys {
		t.Setenv(EnvFor(key), "")
	}
	regMu.Lock()
	prevManagers, prevHost, prevAddr, prevWildcard := managers, baseHost, baseAddr, wildcard
	managers, baseHost, baseAddr, wildcard = nil, "127.0.0.1", []string{"127.0.0.1", "::1"}, false
	regMu.Unlock()
	t.Cleanup(func() {
		regMu.Lock()
		managers, baseHost, baseAddr, wildcard = prevManagers, prevHost, prevAddr, prevWildcard
		regMu.Unlock()
		httpguard.SetTrusted(nil)
		httpguard.SetAccessToken("")
	})
	dbtest.Init(t)
}

func TestParseAddrs(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want []string
		err  string
	}{
		{"empty", "  \n ", []string{}, ""},
		{"tailscale ip", "100.105.22.11", []string{"100.105.22.11"}, ""},
		{"comma newline and dedupe", "100.105.22.11, 100.64.0.1\n100.105.22.11\r\n", []string{"100.105.22.11", "100.64.0.1"}, ""},
		{"ipv6 canonical", "FD7A:115C:A1E0:0:0:0:0:1,[fd7a:115c:a1e0::2]", []string{"fd7a:115c:a1e0::1", "fd7a:115c:a1e0::2"}, ""},
		{"hostname", "100.105.22.11,jasons-macbook-air.ts.net", nil, `"jasons-macbook-air.ts.net" is not an IP address`},
		{"wildcard v4", "0.0.0.0", nil, "is a wildcard"},
		{"wildcard v6", "::", nil, "is a wildcard"},
		{"star", "*", nil, "is not an IP address"},
		{"loopback", "127.0.0.1", nil, "is loopback"},
		{"ipv6 loopback", "::1", nil, "is loopback"},
		{"with port", "100.105.22.11:7830", nil, "is not an IP address"},
		{"leading zero", "100.105.022.11", nil, "is not an IP address"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseAddrs(tt.raw)
			if tt.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.err)
				assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestParseHosts(t *testing.T) {
	long := ""
	for range 64 {
		long += "a"
	}
	tests := []struct {
		name string
		raw  string
		want []string
		bad  bool
	}{
		{"magicdns trailing dot", "Jasons-MacBook-Air.halibut-velociraptor.ts.net.", []string{"jasons-macbook-air.halibut-velociraptor.ts.net"}, false},
		{"short name and ip", "jasons-macbook-air\n100.105.22.11", []string{"jasons-macbook-air", "100.105.22.11"}, false},
		{"dedupe case", "a.ts.net,A.TS.NET", []string{"a.ts.net"}, false},
		{"underscore", "foo_bar.ts.net", nil, true},
		{"leading hyphen", "-foo.ts.net", nil, true},
		{"trailing hyphen", "foo-.ts.net", nil, true},
		{"empty label", "foo..ts.net", nil, true},
		{"wildcard", "*.ts.net", nil, true},
		{"wildcard ip", "0.0.0.0", nil, true},
		{"label too long", long + ".ts.net", nil, true},
		{"url", "http://foo.ts.net", nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseHosts(tt.raw)
			if tt.bad {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "trusted_hosts")
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestNormalizeToken(t *testing.T) {
	got, err := Normalize(KeyAccessToken, "  abc_DEF-123  ")
	require.NoError(t, err)
	assert.Equal(t, "abc_DEF-123", got)
	_, err = Normalize(KeyAccessToken, "two words")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	_, err = Normalize("nope", "x")
	assert.True(t, errs.HasCode(err, errs.CodeNotFound))
}

func TestLoadEnvOverridesSetting(t *testing.T) {
	setupConfig(t)
	require.NoError(t, db.SetSetting(KeyExtraAddrs, "100.64.0.1"))
	require.NoError(t, db.SetSetting(KeyTrustedHosts, "a.ts.net"))
	require.NoError(t, db.SetSetting(KeyAccessToken, "from-setting"))
	cfg := Load()
	assert.Equal(t, []string{"100.64.0.1"}, cfg.ExtraAddrs)
	assert.Equal(t, []string{"a.ts.net"}, cfg.TrustedHosts)
	assert.Equal(t, "from-setting", cfg.Token)
	assert.Empty(t, cfg.Locked)

	t.Setenv(EnvExtraAddrs, "100.64.0.2, bogus-host, 0.0.0.0")
	t.Setenv(EnvAccessToken, "from-env")
	cfg = Load()
	assert.Equal(t, []string{"100.64.0.2"}, cfg.ExtraAddrs, "invalid env entries are skipped")
	assert.Equal(t, "from-env", cfg.Token)
	assert.Equal(t, map[string]string{KeyExtraAddrs: EnvExtraAddrs, KeyAccessToken: EnvAccessToken}, cfg.Locked)
}

func TestSaveValidatesLocksAndApplies(t *testing.T) {
	setupConfig(t)
	err := Save(map[string]string{KeyTrustedHosts: "ok.ts.net", KeyExtraAddrs: "not-an-ip"})
	require.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
	assert.Empty(t, db.GetSetting(KeyTrustedHosts, ""), "nothing stored when any value is rejected")

	require.NoError(t, Save(map[string]string{KeyExtraAddrs: "100.105.22.11\n100.64.0.1", KeyTrustedHosts: "Mac.TS.net.", KeyAccessToken: " tok "}))
	assert.Equal(t, "100.105.22.11,100.64.0.1", db.GetSetting(KeyExtraAddrs, ""))
	assert.Equal(t, "mac.ts.net", db.GetSetting(KeyTrustedHosts, ""))
	assert.Equal(t, "tok", db.GetSetting(KeyAccessToken, ""))
	assert.True(t, httpguard.IsTrustedHost("100.105.22.11:7830"), "extra addresses are trusted")
	assert.True(t, httpguard.IsTrustedHost("mac.ts.net"), "trusted hosts applied live")
	assert.True(t, httpguard.TokenRequired(), "token applied live")

	require.NoError(t, Save(map[string]string{KeyAccessToken: ""}))
	assert.False(t, httpguard.TokenRequired(), "empty token disables auth")

	t.Setenv(EnvTrustedHosts, "env.ts.net")
	err = Save(map[string]string{KeyTrustedHosts: "other.ts.net"})
	require.True(t, errs.HasCode(err, errs.CodeConflict), "%v", err)
	assert.Contains(t, err.Error(), EnvTrustedHosts)
	assert.Equal(t, "mac.ts.net", db.GetSetting(KeyTrustedHosts, ""))

	err = Save(map[string]string{})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

func TestApplyReconcilesRegisteredManagers(t *testing.T) {
	setupConfig(t)
	m := NewManager("mcp", newTestServer(t), 0)
	m.DropGrace = 0
	t.Cleanup(m.Close)
	Register(m)
	require.NoError(t, Save(map[string]string{KeyExtraAddrs: testNet2Addr}))
	st := Listeners()
	require.Len(t, st, 1)
	assert.Equal(t, "mcp", st[0].Server)
	assert.Equal(t, testNet2Addr, st[0].Addr)
	assert.Equal(t, StatusError, st[0].Status, "an address this host doesn't have is reported, not fatal")
	assert.True(t, anyErrors())

	state := CurrentState()
	assert.Equal(t, []string{testNet2Addr}, state.ExtraAddrs)
	assert.False(t, state.TokenSet)
	assert.Equal(t, "127.0.0.1", state.BaseListen)
	assert.Len(t, state.Listeners, 1)

	require.NoError(t, Save(map[string]string{KeyExtraAddrs: ""}))
	assert.Empty(t, Listeners())
}

func TestApplySkipsBaseAndWildcard(t *testing.T) {
	setupConfig(t)
	m := NewManager("mcp", newTestServer(t), 0)
	t.Cleanup(m.Close)
	Register(m)
	SetBase("100.64.0.1")
	require.NoError(t, db.SetSetting(KeyExtraAddrs, "100.64.0.1,"+testNetAddr))
	Apply()
	st := Listeners()
	require.Len(t, st, 1, "the base address itself is skipped")
	assert.Equal(t, testNetAddr, st[0].Addr)
	assert.True(t, httpguard.IsTrustedHost("100.64.0.1"), "a specific non-loopback base is trusted")

	SetBase("0.0.0.0")
	Apply()
	assert.Empty(t, Listeners(), "a wildcard base covers every interface")
	assert.True(t, CurrentState().BaseWildcard)
	assert.True(t, httpguard.IsTrustedHost(testNetAddr), "configured addresses stay trusted for Origin checks")
}
