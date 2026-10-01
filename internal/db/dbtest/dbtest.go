// Package dbtest gives a test its own empty database under a throwaway HOME.
package dbtest

import (
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
// stops before the pools close.
func Init(t testing.TB) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	// DB_PATH takes precedence over HOME; one left set in the developer's shell
	// would point the test at their real database.
	t.Setenv("DB_PATH", "")
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(db.Close)
	return home
}
