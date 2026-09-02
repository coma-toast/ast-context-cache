package db

import (
	"os"
	"testing"
	"time"
)

// TestCompactSkipsWhenAlreadyRunning guards against the bug that caused runaway WAL
// growth and cascading "database is locked" errors: deleting or resetting several
// projects in quick succession each fires its own `go Compact()`, and VACUUM is
// exclusive and expensive — running several concurrently against the same files must
// not happen. A concurrent call has to return immediately (TryLock), not block waiting
// for the first VACUUM to finish.
func TestCompactSkipsWhenAlreadyRunning(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	if !compactMu.TryLock() {
		t.Fatal("expected to acquire compactMu")
	}
	done := make(chan struct{})
	go func() {
		Compact()
		close(done)
	}()
	// Compact() must return promptly rather than block on the held lock.
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Compact() blocked instead of skipping while a VACUUM was already running")
	}
	compactMu.Unlock()

	// A normal call proceeds once nothing else holds the lock.
	Compact()
}
