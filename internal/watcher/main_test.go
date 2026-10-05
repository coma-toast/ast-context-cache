package watcher

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// TestMain opens the db pools once for the whole package. Tests used to point
// HOME somewhere new and call db.Init each, reassigning db's package-global
// pools while goroutines left by earlier tests were still reading them — as is
// idleLoop, which reads settings every 30s for the life of the process.
func TestMain(m *testing.M) {
	home, err := os.MkdirTemp("", "astcache-watcher-home-")
	if err != nil {
		panic(err)
	}
	os.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		panic(err)
	}
	code := m.Run()
	stopWatchersForTest()
	db.Close()
	os.RemoveAll(home)
	os.Exit(code)
}

// cleanupWatchers stops everything t's watchers started once t is done, so
// none of it is still running into the next test. Call it after registering
// any cleanup that restores a package var those goroutines read, so that it
// runs first.
func cleanupWatchers(t *testing.T) {
	t.Helper()
	t.Cleanup(stopWatchersForTest)
}

// stopWatchersForTest deletes every watcher, cancels every pending debounce,
// and waits for the goroutines they started to return.
func stopWatchersForTest() {
	StopAll()
}
