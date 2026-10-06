package httpguard

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// No t.Parallel here: the trusted set and access token are package globals. Go runs
// these sequential tests before releasing the parallel ones above, and each resets the
// globals on cleanup.

const (
	testToken    = "s3cret-token"
	tailnetAddr  = "100.64.0.9:5555"
	loopbackAddr = "127.0.0.1:5555"
)

func withTrusted(t *testing.T, hosts ...string) {
	t.Helper()
	SetTrusted(hosts)
	t.Cleanup(func() { SetTrusted(nil) })
}

func withToken(t *testing.T, token string) {
	t.Helper()
	SetAccessToken(token)
	prev := loginFailureDelay
	loginFailureDelay = 0
	t.Cleanup(func() {
		SetAccessToken("")
		loginFailureDelay = prev
	})
}

func TestTrustedHosts(t *testing.T) {
	withTrusted(t, "100.105.22.11", "Jasons-MacBook-Air.halibut-velociraptor.ts.net.", "fd7a:115c:a1e0::1", "0.0.0.0", "")
	hosts := []struct {
		host string
		want bool
	}{
		{"100.105.22.11", true},
		{"100.105.22.11:7830", true},
		{"jasons-macbook-air.halibut-velociraptor.ts.net:7830", true},
		{"JASONS-MACBOOK-AIR.halibut-velociraptor.ts.net.", true},
		{"[fd7a:115c:a1e0::1]:7821", true},
		{"100.105.22.12:7830", false},
		{"evil.ts.net", false},
		{"0.0.0.0:7830", false},
		{"", false},
	}
	for _, tt := range hosts {
		t.Run("host "+tt.host, func(t *testing.T) {
			assert.Equal(t, tt.want, IsTrustedHost(tt.host))
			assert.Equal(t, tt.want || IsLoopbackHost(tt.host), AllowHost(tt.host, "127.0.0.1"))
		})
	}
	origins := []struct {
		origin string
		want   bool
	}{
		{"http://100.105.22.11:7830", true},
		{"https://jasons-macbook-air.halibut-velociraptor.ts.net", true},
		{"http://[fd7a:115c:a1e0::1]:7830", true},
		{"http://localhost:7830", true},
		{"http://100.105.22.12:7830", false},
		{"http://evil.example", false},
		{"http://halibut-velociraptor.ts.net", false},
	}
	for _, tt := range origins {
		t.Run("origin "+tt.origin, func(t *testing.T) {
			assert.Equal(t, tt.want, AllowOrigin(tt.origin))
		})
	}
}

func TestTrustedHostsReplaced(t *testing.T) {
	withTrusted(t, "100.105.22.11")
	require.True(t, AllowHost("100.105.22.11:7830", "127.0.0.1"))
	SetTrusted([]string{"other.ts.net"})
	assert.False(t, AllowHost("100.105.22.11:7830", "127.0.0.1"), "SetTrusted replaces, not appends")
	assert.True(t, AllowHost("other.ts.net:7830", "127.0.0.1"))
}

func TestMiddlewareTrustedOriginCanWrite(t *testing.T) {
	h := Middleware("127.0.0.1", okHandler())
	req := newRequest(http.MethodPost, "100.105.22.11:7830", "http://100.105.22.11:7830")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	require.Equal(t, http.StatusForbidden, rr.Code, "untrusted until configured")
	withTrusted(t, "100.105.22.11")
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, newRequest(http.MethodPost, "100.105.22.11:7830", "http://100.105.22.11:7830"))
	assert.Equal(t, http.StatusOK, rr.Code)
}

func TestIsLoopbackRemote(t *testing.T) {
	tests := []struct {
		addr string
		want bool
	}{
		{"127.0.0.1:5555", true},
		{"[::1]:5555", true},
		{"[::ffff:127.0.0.1]:5555", true},
		{"127.0.0.1", true},
		{"100.64.0.9:5555", false},
		{"[fe80::1%en0]:5555", false},
		{"", false},
		{"garbage", false},
	}
	for _, tt := range tests {
		t.Run(tt.addr, func(t *testing.T) {
			assert.Equal(t, tt.want, IsLoopbackRemote(tt.addr))
		})
	}
}

func TestTokenEqual(t *testing.T) {
	assert.True(t, tokenEqual(testToken, testToken))
	assert.False(t, tokenEqual(testToken, testToken+"x"), "longer")
	assert.False(t, tokenEqual(testToken[:3], testToken), "prefix")
	assert.False(t, tokenEqual("", testToken))
	assert.False(t, tokenEqual(strings.ToUpper(testToken), testToken))
}

func TestSessionValue(t *testing.T) {
	v := sessionValue(testToken)
	assert.NotContains(t, v, testToken, "cookie never holds the raw token")
	assert.Equal(t, v, sessionValue(testToken), "deterministic")
	assert.NotEqual(t, v, sessionValue(testToken+"2"), "rotating the token changes the cookie")
}

func TestAuthorized(t *testing.T) {
	withToken(t, testToken)
	cookie := &http.Cookie{Name: SessionCookie, Value: sessionValue(testToken)}
	tests := []struct {
		name   string
		remote string
		auth   string
		cookie *http.Cookie
		want   bool
	}{
		{"loopback needs nothing", loopbackAddr, "", nil, true},
		{"ipv6 loopback needs nothing", "[::1]:5555", "", nil, true},
		{"remote without credentials", tailnetAddr, "", nil, false},
		{"remote bad bearer", tailnetAddr, "Bearer wrong", nil, false},
		{"remote good bearer", tailnetAddr, "Bearer " + testToken, nil, true},
		{"remote lowercase scheme", tailnetAddr, "bearer " + testToken, nil, true},
		{"remote basic scheme", tailnetAddr, "Basic " + testToken, nil, false},
		{"remote empty bearer", tailnetAddr, "Bearer ", nil, false},
		{"remote good cookie", tailnetAddr, "", cookie, true},
		{"remote raw token as cookie", tailnetAddr, "", &http.Cookie{Name: SessionCookie, Value: testToken}, false},
		{"remote stale cookie", tailnetAddr, "", &http.Cookie{Name: SessionCookie, Value: sessionValue("old")}, false},
		{"remote bad bearer good cookie", tailnetAddr, "Bearer wrong", cookie, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/x", nil)
			req.RemoteAddr = tt.remote
			if tt.auth != "" {
				req.Header.Set("Authorization", tt.auth)
			}
			if tt.cookie != nil {
				req.AddCookie(tt.cookie)
			}
			assert.Equal(t, tt.want, Authorized(req))
		})
	}
}

func TestAuthorizedWithoutToken(t *testing.T) {
	SetAccessToken("  ")
	assert.False(t, TokenRequired(), "blank token disables auth")
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.RemoteAddr = tailnetAddr
	assert.True(t, Authorized(req))
}

func TestMiddlewareTokenAuth(t *testing.T) {
	withToken(t, testToken)
	withTrusted(t, "100.105.22.11")
	h := Middleware("127.0.0.1", okHandler())
	tests := []struct {
		name     string
		method   string
		path     string
		remote   string
		auth     string
		upgrade  bool
		want     int
		location string
	}{
		{"loopback get", http.MethodGet, "/dashboard/", loopbackAddr, "", false, http.StatusOK, ""},
		{"loopback post", http.MethodPost, "/api/settings", loopbackAddr, "", false, http.StatusOK, ""},
		{"remote page redirects", http.MethodGet, "/dashboard/", tailnetAddr, "", false, http.StatusSeeOther, LoginPath},
		{"remote root redirects", http.MethodGet, "/", tailnetAddr, "", false, http.StatusSeeOther, LoginPath},
		{"remote api get 401", http.MethodGet, "/api/dashboard/settings", tailnetAddr, "", false, http.StatusUnauthorized, ""},
		{"remote metrics 401", http.MethodGet, "/metrics", tailnetAddr, "", false, http.StatusUnauthorized, ""},
		{"remote post 401", http.MethodPost, "/api/settings", tailnetAddr, "", false, http.StatusUnauthorized, ""},
		{"remote websocket 401", http.MethodGet, "/ws", tailnetAddr, "", true, http.StatusUnauthorized, ""},
		{"remote websocket bearer", http.MethodGet, "/ws", tailnetAddr, "Bearer " + testToken, true, http.StatusOK, ""},
		{"remote bad bearer", http.MethodGet, "/api/stats", tailnetAddr, "Bearer nope", false, http.StatusUnauthorized, ""},
		{"remote good bearer", http.MethodPost, "/api/settings", tailnetAddr, "Bearer " + testToken, false, http.StatusOK, ""},
		{"remote login exempt", http.MethodGet, LoginPath, tailnetAddr, "", false, http.StatusOK, ""},
		{"remote login post exempt", http.MethodPost, LoginPath, tailnetAddr, "", false, http.StatusOK, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			req.Host = "100.105.22.11:7830"
			req.RemoteAddr = tt.remote
			if tt.method == http.MethodPost || tt.upgrade {
				req.Header.Set("Origin", "http://100.105.22.11:7830")
			}
			if tt.auth != "" {
				req.Header.Set("Authorization", tt.auth)
			}
			if tt.upgrade {
				req.Header.Set("Connection", "Upgrade")
				req.Header.Set("Upgrade", "websocket")
			}
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			require.Equal(t, tt.want, rr.Code, rr.Body.String())
			if tt.location != "" {
				assert.Equal(t, tt.location, rr.Header().Get("Location"))
			}
			if tt.want == http.StatusUnauthorized {
				assert.JSONEq(t, unauthorizedBody, rr.Body.String())
				assert.Contains(t, rr.Header().Get("WWW-Authenticate"), "Bearer")
			}
		})
	}
}

func TestMiddlewareForeignOriginStillForbiddenWithToken(t *testing.T) {
	withToken(t, testToken)
	h := Middleware("127.0.0.1", okHandler())
	req := newRequest(http.MethodPost, "127.0.0.1:7830", "http://evil.example")
	req.RemoteAddr = tailnetAddr
	req.Header.Set("Authorization", "Bearer "+testToken)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	assert.Equal(t, http.StatusForbidden, rr.Code, "a token doesn't excuse a cross-site write")
}

func TestRequireMCPAuth(t *testing.T) {
	withToken(t, testToken)
	h := RequireMCPAuth(okHandler(), "/health")
	tests := []struct {
		name   string
		method string
		path   string
		remote string
		auth   string
		want   int
	}{
		{"loopback mcp", http.MethodPost, "/mcp", loopbackAddr, "", http.StatusOK},
		{"remote mcp no token", http.MethodPost, "/mcp", tailnetAddr, "", http.StatusUnauthorized},
		{"remote mcp get no token", http.MethodGet, "/mcp", tailnetAddr, "", http.StatusUnauthorized},
		{"remote mcp bad token", http.MethodPost, "/mcp", tailnetAddr, "Bearer wrong", http.StatusUnauthorized},
		{"remote mcp good token", http.MethodPost, "/mcp", tailnetAddr, "Bearer " + testToken, http.StatusOK},
		{"remote embed no token", http.MethodPost, "/embed", tailnetAddr, "", http.StatusUnauthorized},
		{"remote root no token", http.MethodGet, "/", tailnetAddr, "", http.StatusUnauthorized},
		{"remote health exempt", http.MethodGet, "/health", tailnetAddr, "", http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			req.RemoteAddr = tt.remote
			if tt.auth != "" {
				req.Header.Set("Authorization", tt.auth)
			}
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			require.Equal(t, tt.want, rr.Code)
			if tt.want == http.StatusUnauthorized {
				assert.JSONEq(t, `{"jsonrpc":"2.0","error":{"code":-32001,"message":"unauthorized"}}`, rr.Body.String())
			}
		})
	}
}

func TestHandleLogin(t *testing.T) {
	withToken(t, testToken)
	post := func(token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, LoginPath, strings.NewReader(url.Values{"token": {token}}.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.RemoteAddr = tailnetAddr
		rr := httptest.NewRecorder()
		HandleLogin(rr, req)
		return rr
	}
	t.Run("form", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, LoginPath, nil)
		req.RemoteAddr = tailnetAddr
		rr := httptest.NewRecorder()
		HandleLogin(rr, req)
		require.Equal(t, http.StatusOK, rr.Code)
		assert.Contains(t, rr.Header().Get("Content-Type"), "text/html")
		assert.Contains(t, rr.Body.String(), `name="token"`)
		assert.Contains(t, rr.Body.String(), `method="post"`)
		assert.NotContains(t, rr.Body.String(), "http://", "no external assets")
	})
	t.Run("wrong token", func(t *testing.T) {
		rr := post("wrong")
		require.Equal(t, http.StatusUnauthorized, rr.Code)
		assert.Contains(t, rr.Body.String(), loginWrongToken)
		assert.Empty(t, rr.Result().Cookies())
	})
	t.Run("good token", func(t *testing.T) {
		rr := post(testToken)
		require.Equal(t, http.StatusSeeOther, rr.Code)
		assert.Equal(t, dashboardHome, rr.Header().Get("Location"))
		cookies := rr.Result().Cookies()
		require.Len(t, cookies, 1)
		c := cookies[0]
		assert.Equal(t, SessionCookie, c.Name)
		assert.Equal(t, sessionValue(testToken), c.Value)
		assert.NotContains(t, c.Value, testToken)
		assert.True(t, c.HttpOnly)
		assert.False(t, c.Secure, "plain HTTP")
		assert.Equal(t, http.SameSiteStrictMode, c.SameSite)
		assert.Equal(t, "/", c.Path)
		next := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
		next.RemoteAddr = tailnetAddr
		next.AddCookie(c)
		assert.True(t, Authorized(next), "the cookie authenticates later requests")
		SetAccessToken("rotated")
		assert.False(t, Authorized(next), "rotating the token invalidates the cookie")
	})
	t.Run("already authorized redirects", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, LoginPath, nil)
		req.RemoteAddr = loopbackAddr
		rr := httptest.NewRecorder()
		HandleLogin(rr, req)
		assert.Equal(t, http.StatusSeeOther, rr.Code)
	})
}

func TestHandleLoginWithoutToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, LoginPath, strings.NewReader("token=x"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.RemoteAddr = tailnetAddr
	rr := httptest.NewRecorder()
	HandleLogin(rr, req)
	assert.Equal(t, http.StatusSeeOther, rr.Code)
	assert.Empty(t, rr.Result().Cookies())
}
