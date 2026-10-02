package mcp

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// TestMain points HOME at a throwaway directory for the whole binary (so no test can
// reach the real ~/.astcache or ~/.astcache.location) and opens the databases once.
// Re-running db.Init per test races with the index writer goroutine of the previous
// Init under -race, so tests that only need a working DB rely on this one.
func TestMain(m *testing.M) {
	home, err := os.MkdirTemp("", "astcache-mcp-home-")
	if err != nil {
		panic(err)
	}
	os.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		panic(err)
	}
	code := m.Run()
	// No db.Close: watchers started by handleToolCall may still be writing, and closing
	// the pools under them is a (harmless, teardown-only) race. The process exits next.
	os.RemoveAll(home)
	os.Exit(code)
}
