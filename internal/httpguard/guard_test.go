package httpguard

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsLoopbackHost(t *testing.T) {
	t.Parallel()
	tests := []struct {
		host string
		want bool
	}{
		{"localhost", true},
		{"LOCALHOST", true},
		{"localhost:7830", true},
		{"localhost.", true},
		{"app.localhost:3000", true},
		{"127.0.0.1", true},
		{"127.0.0.1:7821", true},
		{"127.8.9.10", true},
		{"::1", true},
		{"[::1]", true},
		{"[::1]:7821", true},
		{"", false},
		{"0.0.0.0", false},
		{"192.168.1.10:7830", false},
		{"evil.example", false},
		{"localhost.evil.example", false},
		{"127.0.0.1.evil.example", false},
		{"[::2]:7821", false},
	}
	for _, tt := range tests {
		t.Run(tt.host, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, IsLoopbackHost(tt.host))
		})
	}
}

func TestAllowOrigin(t *testing.T) {
	t.Parallel()
	tests := []struct {
		origin string
		want   bool
	}{
		{"", true},
		{"http://localhost:7830", true},
		{"http://127.0.0.1:7830", true},
		{"https://127.0.0.1", true},
		{"http://[::1]:7830", true},
		{"http://dev.localhost:5173", true},
		{"http://evil.example", false},
		{"https://localhost.evil.example", false},
		{"http://192.168.1.10:7830", false},
		{"null", false},
		{"chrome-extension://abcdef", false},
		{"://bad", false},
	}
	for _, tt := range tests {
		t.Run(tt.origin, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, AllowOrigin(tt.origin))
		})
	}
}

func TestAllowHost(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		host   string
		listen string
		want   bool
	}{
		{"loopback host loopback listen", "127.0.0.1:7821", "127.0.0.1", true},
		{"localhost host loopback listen", "localhost:7830", "127.0.0.1", true},
		{"ipv6 loopback host", "[::1]:7821", "127.0.0.1", true},
		{"foreign host loopback listen", "evil.example:7830", "127.0.0.1", false},
		{"empty host loopback listen", "", "127.0.0.1", false},
		{"any host wildcard v4", "myserver.lan:7830", "0.0.0.0", true},
		{"any host wildcard v6", "myserver.lan:7830", "::", true},
		{"any host empty listen", "myserver.lan:7830", "", true},
		{"listen ip matches", "100.64.1.2:7830", "100.64.1.2", true},
		{"listen ip loopback still ok", "localhost:7830", "100.64.1.2", true},
		{"listen ip other host", "evil.example:7830", "100.64.1.2", false},
		{"listen ipv6 matches", "[fd00::1]:7821", "fd00::1", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, AllowHost(tt.host, tt.listen))
		})
	}
}

func TestMiddleware(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		method  string
		host    string
		origin  string
		upgrade bool
		want    int
	}{
		{"post foreign origin", http.MethodPost, "127.0.0.1:7830", "http://evil.example", false, http.StatusForbidden},
		{"post loopback origin", http.MethodPost, "127.0.0.1:7830", "http://127.0.0.1:7830", false, http.StatusOK},
		{"post no origin", http.MethodPost, "127.0.0.1:7830", "", false, http.StatusOK},
		{"delete foreign origin", http.MethodDelete, "127.0.0.1:7830", "http://evil.example", false, http.StatusForbidden},
		{"get foreign origin", http.MethodGet, "127.0.0.1:7830", "http://evil.example", false, http.StatusOK},
		{"get foreign host", http.MethodGet, "evil.example:7830", "", false, http.StatusOK},
		{"websocket foreign origin", http.MethodGet, "127.0.0.1:7830", "http://evil.example", true, http.StatusForbidden},
		{"websocket loopback origin", http.MethodGet, "localhost:7830", "http://localhost:7830", true, http.StatusOK},
		{"post rebinding host", http.MethodPost, "evil.example:7830", "", false, http.StatusForbidden},
		{"websocket rebinding host", http.MethodGet, "evil.example:7830", "", true, http.StatusForbidden},
	}
	h := Middleware("127.0.0.1", okHandler())
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			req := newRequest(tt.method, tt.host, tt.origin)
			if tt.upgrade {
				req.Header.Set("Connection", "Upgrade")
				req.Header.Set("Upgrade", "websocket")
			}
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			assert.Equal(t, tt.want, rr.Code)
			if tt.want == http.StatusForbidden {
				assert.Equal(t, "application/json", rr.Header().Get("Content-Type"))
				assert.Contains(t, rr.Body.String(), "forbidden")
			}
		})
	}
}

func TestMCPMiddleware(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		method string
		host   string
		origin string
		want   int
	}{
		{"post foreign origin", http.MethodPost, "127.0.0.1:7821", "http://evil.example", http.StatusForbidden},
		{"get foreign origin", http.MethodGet, "127.0.0.1:7821", "http://evil.example", http.StatusForbidden},
		{"delete foreign origin", http.MethodDelete, "127.0.0.1:7821", "http://evil.example", http.StatusForbidden},
		{"get rebinding host", http.MethodGet, "evil.example:7821", "", http.StatusForbidden},
		{"post no origin", http.MethodPost, "127.0.0.1:7821", "", http.StatusOK},
		{"post loopback origin", http.MethodPost, "localhost:7821", "http://localhost:6274", http.StatusOK},
	}
	h := MCPMiddleware("127.0.0.1", okHandler())
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, newRequest(tt.method, tt.host, tt.origin))
			require.Equal(t, tt.want, rr.Code)
			if tt.want == http.StatusForbidden {
				assert.JSONEq(t, `{"jsonrpc":"2.0","error":{"code":-32600,"message":"`+forbiddenMessage(tt.origin)+`"}}`, rr.Body.String())
			}
		})
	}
}

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
}

func newRequest(method, host, origin string) *http.Request {
	req := httptest.NewRequest(method, "/x", nil)
	req.Host = host
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	return req
}

func forbiddenMessage(origin string) string {
	if origin != "" {
		return "forbidden origin"
	}
	return "forbidden host"
}
