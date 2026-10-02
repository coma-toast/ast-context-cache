// Package dbtest gives a test its own empty database under a throwaway HOME.
package dbtest

import (
	"sync"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Init points HOME at a fresh t.TempDir(), opens the db pools there, and closes
// them when t finishes. It returns the new HOME.
//
// The pools are package globals. A test that opened them and never closed them
// left every later test in the binary reading and writing its TempDir: once
// that was removed, they failed with "unable to open database file" (or passed
// only because an earlier test happened to have opened a database for them),
// and anything still writing through the pools could recreate .astcache while
// t.TempDir's cleanup was removing it ("directory not empty").
//
// Cleanups run last-registered first, so Close runs before the TempDir is
// removed. A test that starts anything else using the pools (a watcher, a
// background refresh) must register its stop after calling Init, so that it
// stops before the pools close — or the package registers a WaitFor hook.
func Init(t testing.TB) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	// DB_PATH takes precedence over HOME; one left set in the developer's shell
	// would point the test at their real database.
	t.Setenv("DB_PATH", "")
	runWaits()
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		runWaits()
		db.Close()
	})
	return home
}

var (
	waitsMu sync.Mutex
	waits   []func()
)

// WaitFor registers fn to run before every Init reassigns the pools and before
// its cleanup closes them. fn should wait for the package's own background
// goroutines that read the pools, which a test can start without knowing it
// (an HTTP handler refreshing a cache, say). Running it before Init too covers
// goroutines left by a test that never opened a database itself. Call it from
// an init func in a _test.go file.
func WaitFor(fn func()) {
	waitsMu.Lock()
	defer waitsMu.Unlock()
	waits = append(waits, fn)
}

func runWaits() {
	waitsMu.Lock()
	fns := append([]func(){}, waits...)
	waitsMu.Unlock()
	for _, fn := range fns {
		fn()
	}
}
