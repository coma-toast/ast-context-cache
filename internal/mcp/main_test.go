package mcp

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// handleToolCall starts a watcher on project_path, and a watcher's goroutines
// (debounced re-indexes, catch-ups) read the db pools. Stop them all before
// dbtest reassigns or closes the pools, or they race with the teardown.
func init() {
	dbtest.WaitFor(watcher.StopAll)
}

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
	watcher.StopAll()
	db.Close()
	os.RemoveAll(home)
	os.Exit(code)
}
