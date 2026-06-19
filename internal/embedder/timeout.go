package embedder

import (
	"strings"
	"time"
)

// DefaultRemoteTimeout is the HTTP client timeout for dashboard probes and connectivity checks.
const DefaultRemoteTimeout = 15 * time.Second

// DefaultHTTPEmbedTimeout is the HTTP client timeout for remote embedding API calls.
const DefaultHTTPEmbedTimeout = 300 * time.Second

// ResolveRemoteTimeout returns EMBED_REMOTE_TIMEOUT when set (env or dashboard settings),
// otherwise DefaultHTTPEmbedTimeout.
func ResolveRemoteTimeout() time.Duration {
	if d, ok := parseDurationSetting(EffectiveEnv("EMBED_REMOTE_TIMEOUT")); ok {
		return d
	}
	return DefaultHTTPEmbedTimeout
}

func parseDurationSetting(v string) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 0, false
	}
	return d, true
}
