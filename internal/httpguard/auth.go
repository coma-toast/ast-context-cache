package httpguard

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"html/template"
	"net"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"time"
)

const (
	// LoginPath serves the dashboard login form (GET) and accepts the access token (POST).
	LoginPath = "/login"
	// SessionCookie carries an HMAC of the access token for dashboard browsers, never the token.
	SessionCookie = "astcache_session"

	sessionMessage      = "astcache_session/v1"
	sessionMaxAge       = 30 * 24 * time.Hour
	loginTokenField     = "token"
	loginMaxBodyBytes   = 4 << 10
	dashboardHome       = "/dashboard/"
	unauthorizedBody    = `{"error":"unauthorized"}`
	mcpUnauthorizedBody = `{"jsonrpc":"2.0","error":{"code":-32001,"message":"unauthorized"}}`
	loginWrongToken     = "That token is not valid."
)

// tokenState is the configured access token and the session cookie value derived from it.
type tokenState struct {
	token   string
	session string
}

// access is nil while no token is configured, which leaves every client unauthenticated.
var access atomic.Pointer[tokenState]

// loginFailureDelay slows down guessing through the login form; tests set it to zero.
var loginFailureDelay = 500 * time.Millisecond

var loginPage = template.Must(template.New("login").Parse(loginHTML))

// SetAccessToken sets the token non-loopback clients must present; "" disables the
// requirement. Rotating it invalidates every session cookie issued for the old token.
func SetAccessToken(token string) {
	token = strings.TrimSpace(token)
	if token == "" {
		access.Store(nil)
		return
	}
	access.Store(&tokenState{token: token, session: sessionValue(token)})
}

// TokenRequired reports whether an access token is configured.
func TokenRequired() bool {
	return access.Load() != nil
}

// IsLoopbackRemote reports whether remoteAddr (an http.Request.RemoteAddr) is a
// loopback IP. Anything unparseable counts as remote, so it must authenticate.
func IsLoopbackRemote(remoteAddr string) bool {
	h, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		h = remoteAddr
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// Authorized reports whether r may proceed: no token is configured, r comes from a
// loopback address, or it carries "Authorization: Bearer <token>" or a valid
// session cookie.
func Authorized(r *http.Request) bool {
	st := access.Load()
	if st == nil || IsLoopbackRemote(r.RemoteAddr) {
		return true
	}
	if tok, ok := bearerToken(r); ok && tokenEqual(tok, st.token) {
		return true
	}
	c, err := r.Cookie(SessionCookie)
	return err == nil && tokenEqual(c.Value, st.session)
}

// RequireMCPAuth enforces the access token on every path of the MCP port except
// exempt ones (the health probe), answering a JSON-RPC 401 with code -32001.
func RequireMCPAuth(next http.Handler, exempt ...string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !slices.Contains(exempt, r.URL.Path) && !Authorized(r) {
			logger.Warn("Rejected unauthenticated MCP request", "method", r.Method, "path", r.URL.Path, "remote", r.RemoteAddr)
			writeUnauthorized(w, mcpUnauthorizedBody)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// HandleLogin serves the dashboard login form. A POST with the right token sets the
// session cookie and redirects to the dashboard; a wrong one re-renders the form
// after a short delay. Without a configured token, or for a client that is already
// authorized, both methods just redirect to the dashboard.
func HandleLogin(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet, http.MethodHead:
		if Authorized(r) {
			http.Redirect(w, r, dashboardHome, http.StatusSeeOther)
			return
		}
		renderLogin(w, http.StatusOK, "")
	case http.MethodPost:
		postLogin(w, r)
	default:
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func postLogin(w http.ResponseWriter, r *http.Request) {
	st := access.Load()
	if st == nil {
		http.Redirect(w, r, dashboardHome, http.StatusSeeOther)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, loginMaxBodyBytes)
	if err := r.ParseForm(); err != nil || !tokenEqual(strings.TrimSpace(r.PostForm.Get(loginTokenField)), st.token) {
		logger.Warn("Rejected dashboard login", "remote", r.RemoteAddr)
		time.Sleep(loginFailureDelay)
		renderLogin(w, http.StatusUnauthorized, loginWrongToken)
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     SessionCookie,
		Value:    st.session,
		Path:     "/",
		MaxAge:   int(sessionMaxAge.Seconds()),
		HttpOnly: true,
		Secure:   r.TLS != nil,
		SameSite: http.SameSiteStrictMode,
	})
	logger.Info("Dashboard login", "remote", r.RemoteAddr)
	http.Redirect(w, r, dashboardHome, http.StatusSeeOther)
}

func renderLogin(w http.ResponseWriter, status int, errMsg string) {
	h := w.Header()
	h.Set("Content-Type", "text/html; charset=utf-8")
	h.Set("Cache-Control", "no-store")
	h.Set("X-Frame-Options", "DENY")
	h.Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
	w.WriteHeader(status)
	if err := loginPage.Execute(w, map[string]string{"Error": errMsg, "Field": loginTokenField, "Action": LoginPath}); err != nil {
		logger.Warn("Failed to render login page", "error", err)
	}
}

// denyDashboard answers an unauthenticated dashboard request: browsers navigating to
// a page are sent to the login form, while API, WebSocket and metrics calls get 401.
func denyDashboard(w http.ResponseWriter, r *http.Request) {
	if wantsHTML(r) {
		http.Redirect(w, r, LoginPath, http.StatusSeeOther)
		return
	}
	logger.Warn("Rejected unauthenticated dashboard request", "method", r.Method, "path", r.URL.Path, "remote", r.RemoteAddr)
	writeUnauthorized(w, unauthorizedBody)
}

func wantsHTML(r *http.Request) bool {
	if (r.Method != http.MethodGet && r.Method != http.MethodHead) || isStateChanging(r) {
		return false
	}
	p := r.URL.Path
	return !strings.HasPrefix(p, "/api/") && p != "/ws" && p != "/metrics"
}

func writeUnauthorized(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("WWW-Authenticate", `Bearer realm="ast-context-cache"`)
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(body))
}

func bearerToken(r *http.Request) (string, bool) {
	scheme, tok, ok := strings.Cut(r.Header.Get("Authorization"), " ")
	if !ok || !strings.EqualFold(scheme, "bearer") {
		return "", false
	}
	tok = strings.TrimSpace(tok)
	return tok, tok != ""
}

// sessionValue derives the cookie value from token, so the cookie never holds the
// token itself and changes whenever the token does.
func sessionValue(token string) string {
	mac := hmac.New(sha256.New, []byte(token))
	mac.Write([]byte(sessionMessage))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

// tokenEqual compares in constant time. Hashing both sides first gives equal-length
// inputs, so the time taken doesn't reveal the expected token's length either.
func tokenEqual(got, want string) bool {
	a, b := sha256.Sum256([]byte(got)), sha256.Sum256([]byte(want))
	return subtle.ConstantTimeCompare(a[:], b[:]) == 1
}

const loginHTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>ast-context-cache login</title>
<style>
:root { color-scheme: light dark; --bg: #f6f8fa; --card: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0; --accent: #0969da; --err: #d1242f; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --card: #161b22; --fg: #e6edf3; --muted: #9198a1; --border: #30363d; --accent: #4493f8; --err: #f85149; } }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 16px; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
form { width: 100%; max-width: 380px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 24px; }
h1 { margin: 0 0 4px; font-size: 18px; }
p { margin: 0 0 16px; color: var(--muted); font-size: 13px; }
label { display: block; font-size: 13px; margin-bottom: 6px; }
input { width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--fg); font: inherit; }
button { margin-top: 16px; width: 100%; padding: 8px 12px; border: 0; border-radius: 6px; background: var(--accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
.err { color: var(--err); margin: 0 0 12px; font-size: 13px; }
</style>
</head>
<body>
<form method="post" action="{{.Action}}">
<h1>ast-context-cache</h1>
<p>This server requires an access token for remote connections.</p>
{{if .Error}}<div class="err" role="alert">{{.Error}}</div>{{end}}
<label for="token">Access token</label>
<input id="token" name="{{.Field}}" type="password" autocomplete="current-password" autofocus required>
<button type="submit">Sign in</button>
</form>
</body>
</html>
`
