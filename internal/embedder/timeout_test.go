package embedder

import (
	"testing"
	"time"
)

func TestResolveRemoteTimeout_default(t *testing.T) {
	t.Setenv("EMBED_REMOTE_TIMEOUT", "")
	if got := ResolveRemoteTimeout(); got != DefaultHTTPEmbedTimeout {
		t.Fatalf("got %s want %s", got, DefaultHTTPEmbedTimeout)
	}
}

func TestResolveRemoteTimeout_env(t *testing.T) {
	t.Setenv("EMBED_REMOTE_TIMEOUT", "45s")
	if got := ResolveRemoteTimeout(); got != 45*time.Second {
		t.Fatalf("got %s", got)
	}
}
