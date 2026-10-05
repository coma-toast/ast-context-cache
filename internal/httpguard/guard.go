// Package httpguard rejects cross-origin and DNS-rebinding requests to the local
// MCP and dashboard servers. Both servers have no authentication, so a browser tab
// on any website could otherwise drive them: a foreign Origin header catches plain
// cross-site requests, and a Host check catches DNS rebinding, where the attacker's
// hostname resolves to 127.0.0.1 and the browser treats the request as same-origin.
package httpguard

import (
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/logging"
)

const (
	forbiddenOriginBody    = `{"error":"forbidden origin"}`
	forbiddenHostBody      = `{"error":"forbidden host"}`
	mcpForbiddenOriginBody = `{"jsonrpc":"2.0","error":{"code":-32600,"message":"forbidden origin"}}`
	mcpForbiddenHostBody   = `{"jsonrpc":"2.0","error":{"code":-32600,"message":"forbidden host"}}`
)

var logger = logging.Tagged("httpguard")

// IsLoopbackHost reports whether host names this machine: localhost, *.localhost,
// 127.0.0.0/8 or ::1. A port and IPv6 brackets are stripped first.
func IsLoopbackHost(host string) bool {
	h := hostOnly(host)
	if h == "localhost" || strings.HasSuffix(h, ".localhost") {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// AllowOrigin reports whether a request carrying this Origin header may act on the
// server. A missing Origin is allowed because native MCP clients, CLIs and curl
// don't send one, while browsers always do on cross-origin writes. Otherwise only
// loopback origins (any port, which covers the dashboard UI) are allowed.
func AllowOrigin(origin string) bool {
	if origin == "" {
		return true
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	return IsLoopbackHost(u.Hostname())
}

// AllowHost reports whether the Host header is one the server expects given the
// address it listens on. Loopback names are always allowed. A wildcard listen
// address ("", 0.0.0.0, ::), as in Docker, accepts any Host because the server is
// deliberately reachable under names it can't know; Origin checks still apply.
// A specific listen address additionally allows that address itself.
func AllowHost(host, listen string) bool {
	if IsLoopbackHost(host) {
		return true
	}
	l := hostOnly(listen)
	if isWildcard(l) {
		return true
	}
	h := hostOnly(host)
	return h != "" && h == l
}

// Middleware guards the dashboard: writes (POST, PUT, PATCH, DELETE) and WebSocket
// upgrades are rejected with 403 when the Origin is foreign or the Host isn't
// expected for listen. Plain GET and HEAD pass through, so read-only scrapers
// such as Prometheus keep working under any Host.
func Middleware(listen string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isStateChanging(r) {
			next.ServeHTTP(w, r)
			return
		}
		if reject(w, r, listen, forbiddenOriginBody, forbiddenHostBody) {
			return
		}
		next.ServeHTTP(w, r)
	})
}

// MCPMiddleware guards the Streamable HTTP /mcp endpoint, applying the Origin and
// Host checks to every method: the MCP spec requires Origin validation on all
// incoming connections and a 403 for an invalid one. The body is a JSON-RPC error
// without an id, as the spec allows.
func MCPMiddleware(listen string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if reject(w, r, listen, mcpForbiddenOriginBody, mcpForbiddenHostBody) {
			return
		}
		next.ServeHTTP(w, r)
	})
}

// reject writes a 403 and returns true when r fails the Origin or Host check.
func reject(w http.ResponseWriter, r *http.Request, listen, originBody, hostBody string) bool {
	origin := r.Header.Get("Origin")
	switch {
	case !AllowOrigin(origin):
		logger.Warn("Rejected request with foreign Origin", "method", r.Method, "path", r.URL.Path, "origin", origin, "host", r.Host)
		writeForbidden(w, originBody)
		return true
	case !AllowHost(r.Host, listen):
		logger.Warn("Rejected request with unexpected Host", "method", r.Method, "path", r.URL.Path, "host", r.Host, "listen", listen)
		writeForbidden(w, hostBody)
		return true
	}
	return false
}

func writeForbidden(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusForbidden)
	_, _ = w.Write([]byte(body))
}

func isStateChanging(r *http.Request) bool {
	switch r.Method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	}
	return strings.Contains(strings.ToLower(r.Header.Get("Upgrade")), "websocket")
}

// hostOnly strips any port and IPv6 brackets and lowercases host, so "[::1]:7821",
// "::1" and "LOCALHOST." compare as their bare names.
func hostOnly(host string) string {
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	return strings.TrimSuffix(strings.ToLower(host), ".")
}

func isWildcard(host string) bool {
	if host == "" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsUnspecified()
}
