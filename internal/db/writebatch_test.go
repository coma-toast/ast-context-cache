package db

import (
	"testing"
	"time"
)

// Every Init started two more flush goroutines that nothing stopped, and a full
// buffer spawned a flush goroutine of its own. Both read DB while the next Init
// reassigned it — a data race under -race, seen from the watcher tests, each of
// which calls Init. Init and Close now stop the batchers before touching the
// pools, and a full buffer kicks a batcher instead of spawning a goroutine.
func TestReinitDoesNotRaceWriteBatchers(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Cleanup(Close)
	for i := 0; i < 3; i++ {
		if err := Init(); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < queryLogFlushSize; j++ {
			LogQuery("test_tool", nil, QueryLogMetrics{}, "/p", "")
		}
		if i == 0 {
			waitQueryRows(t, queryLogFlushSize)
		}
	}
	Close()
	batcherMu.Lock()
	running := batcherStop != nil
	batcherMu.Unlock()
	if running {
		t.Fatal("write batchers still running after Close")
	}
}

// waitQueryRows waits for a full buffer's kick to flush it, well before the
// batcher's periodic tick would.
func waitQueryRows(t *testing.T, want int) {
	t.Helper()
	deadline := time.Now().Add(queryLogFlushInterval / 2)
	var n int
	for time.Now().Before(deadline) {
		if err := DB.QueryRow(`SELECT COUNT(*) FROM queries`).Scan(&n); err == nil && n >= want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("full query buffer not flushed: %d of %d rows", n, want)
}
