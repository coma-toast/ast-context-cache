package dashboard

import (
	"testing"
	"time"
)

func TestLogViewOptsFastDefaults(t *testing.T) {
	cachedLogOptsMu.Lock()
	logOptsLoaded = false
	cachedLogOptsMu.Unlock()
	opts := logViewOptsFast()
	if opts.TailLines != 200 || opts.MaxLineChars != 500 {
		t.Fatalf("defaults=%+v", opts)
	}
}

func TestClampInt(t *testing.T) {
	if clampInt(10, 50, 500) != 50 {
		t.Fatal("clamp low")
	}
	if clampInt(1000, 50, 500) != 500 {
		t.Fatal("clamp high")
	}
}

func TestRecentQueryTimeoutConstant(t *testing.T) {
	if recentQueryTimeout < 3*time.Second {
		t.Fatal("recent query timeout too short")
	}
}
