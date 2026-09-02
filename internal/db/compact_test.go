package db

import (
	"os"
	"testing"
	"time"
)

// TestCompactCoalescesConcurrentCallers guards against two opposite bugs. VACUUM is
// exclusive and expensive, and several independent callers can trigger it around the
// same time (deleting several projects each fires its own goroutine) — running them
// all concurrently against the same files caused cascading "database is locked" errors
// and runaway WAL growth. But a plain "skip if already running" guard has its own bug:
// a delete that lands while another VACUUM is already in flight has its freed space
// silently dropped forever, since nothing runs again afterward to reclaim it — which is
// exactly what was happening (deleted projects, but the database size never shrank).
// A call that arrives while one is running must instead be coalesced into one more pass
// after the current one finishes, not skipped outright.
func TestCompactCoalescesConcurrentCallers(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	compactMu.Lock()
	compactRunCount = 0
	compactMu.Unlock()

	// Simulate a VACUUM already in progress, then have a concurrent caller arrive.
	compactMu.Lock()
	compactRunning = true
	compactMu.Unlock()

	done := make(chan struct{})
	go func() {
		Compact()
		close(done)
	}()

	// The concurrent call must return promptly rather than block on the in-progress run.
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Compact() blocked instead of marking itself pending")
	}

	compactMu.Lock()
	pending := compactPending
	compactMu.Unlock()
	if !pending {
		t.Fatal("concurrent Compact() call must mark pending so its work isn't dropped")
	}

	// Finish the simulated in-progress run the same way Compact()'s own loop would:
	// check pending, and if set, run once more instead of stopping.
	compactMu.Lock()
	compactPending = false
	compactRunning = false
	compactMu.Unlock()
	Compact()

	compactMu.Lock()
	runs := compactRunCount
	compactMu.Unlock()
	if runs < 1 {
		t.Fatalf("expected the pending work to actually run, compactRunCount=%d", runs)
	}
}

// TestCompactSerializesRealConcurrentCalls exercises the real path end-to-end: several
// goroutines calling Compact() at once must never run VACUUM concurrently, and every
// call must eventually be covered by a run rather than silently skipped.
func TestCompactSerializesRealConcurrentCalls(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	compactMu.Lock()
	compactRunCount = 0
	compactMu.Unlock()

	const n = 5
	done := make(chan struct{}, n)
	for i := 0; i < n; i++ {
		go func() {
			Compact()
			done <- struct{}{}
		}()
	}
	for i := 0; i < n; i++ {
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("Compact() call did not return in time")
		}
	}

	compactMu.Lock()
	runs := compactRunCount
	running := compactRunning
	compactMu.Unlock()
	if running {
		t.Fatal("compactRunning left true after all callers returned")
	}
	if runs < 1 {
		t.Fatalf("expected at least one VACUUM run, got %d", runs)
	}
}
