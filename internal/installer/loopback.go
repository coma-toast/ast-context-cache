package installer

import (
	"encoding/json"
	"net"
	"net/url"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/httpguard"
)

// loopbackHost is the one spelling loopback MCP URLs are rewritten to before comparing entries.
const loopbackHost = "127.0.0.1"

// entryHashes fingerprints a server entry found on disk. norm, with loopback MCP URLs rewritten
// to one spelling, is what status and plans compare: http://localhost:7821/mcp (what mcp-local
// registers) and http://127.0.0.1:7821/mcp (the installer default) are the same server. raw is
// the plain hash that installs before this normalization recorded in installer_state.
type entryHashes struct {
	norm string
	raw  string
}

func hashEntry(canon []byte) entryHashes {
	return entryHashes{norm: shortHash(normalizeLoopbackURLs(canon)), raw: shortHash(canon)}
}

// desiredEntryHash is the comparable hash of the entry the installer would write.
func desiredEntryHash(canon []byte) string {
	return shortHash(normalizeLoopbackURLs(canon))
}

// normalizeLoopbackURLs rewrites every http(s) URL in a canonical JSON document whose host is a
// loopback name (localhost, 127.0.0.1, [::1]) to host 127.0.0.1, keeping scheme, port, path and
// query. Input that isn't JSON is returned unchanged.
func normalizeLoopbackURLs(canon []byte) []byte {
	var v any
	if err := json.Unmarshal(canon, &v); err != nil {
		return canon
	}
	out, err := marshalNoEscape(rewriteLoopbackURLs(v))
	if err != nil {
		return canon
	}
	return out
}

func rewriteLoopbackURLs(v any) any {
	switch x := v.(type) {
	case string:
		return loopbackURL(x)
	case []any:
		for i := range x {
			x[i] = rewriteLoopbackURLs(x[i])
		}
		return x
	case map[string]any:
		for k, e := range x {
			x[k] = rewriteLoopbackURLs(e)
		}
		return x
	default:
		return v
	}
}

// loopbackURL returns s with a loopback host rewritten to 127.0.0.1, or s unchanged when it isn't
// an http(s) URL on a loopback host.
func loopbackURL(s string) string {
	if !strings.HasPrefix(s, "http://") && !strings.HasPrefix(s, "https://") {
		return s
	}
	u, err := url.Parse(s)
	if err != nil || !httpguard.IsLoopbackHost(u.Hostname()) {
		return s
	}
	port := u.Port()
	u.Host = loopbackHost
	if port != "" {
		u.Host = net.JoinHostPort(loopbackHost, port)
	}
	return u.String()
}
