package embedqueue

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// TestMain opens the db pools once, for every test in the package. Start's
// goroutines run for the rest of the process and read those pools, so a test
// calling db.Init again after any test has called Start races them.
func TestMain(m *testing.M) {
	if db.IndexDB != nil {
		db.Close()
	}
	tmpHome, err := os.MkdirTemp("", "astcache-embedqueue-home-")
	if err != nil {
		panic(err)
	}
	os.Setenv("HOME", tmpHome)
	if err := db.Init(); err != nil {
		panic(err)
	}
	code := m.Run()
	db.Close()
	os.RemoveAll(tmpHome)
	os.Exit(code)
}

// resetLanesForTest swaps in fresh job channels of the given size and empty pending
// state, and empties both again after the test so its leftovers can't reach the next
// one's workers. It waits for every worker to exit first: one still running from an
// earlier test would take the jobs this test queues.
func resetLanesForTest(t *testing.T, size int) laneSet {
	t.Helper()
	waitLiveZero(t, &workerLive)
	waitLiveZero(t, &auxWorkerLive)
	setLanes(make(chan job, size), make(chan job, size), make(chan job, size))
	clearPendingForTest()
	l := queueLanes()
	t.Cleanup(func() {
		for _, ch := range []chan job{l.pending, l.high, l.low} {
			drainForTest(ch)
		}
		clearPendingForTest()
	})
	return l
}

func drainForTest(ch chan job) {
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}

func clearPendingForTest() {
	pendingMu.Lock()
	pending = map[string]job{}
	pendingChQueued = map[string]struct{}{}
	pendingMu.Unlock()
}

// stopWorkersForTest stops the primary pool a test started and waits for every
// worker goroutine to exit, so later tests can replace the state workers read.
// SetWorkerCount(0) starts a quiet-on-pause goroutine, which the second reset ends.
func stopWorkersForTest(t *testing.T) {
	t.Helper()
	resetPauseStateForTest()
	_, _ = SetWorkerCount(0)
	resetPauseStateForTest()
	waitLiveZero(t, &workerLive)
	waitLiveZero(t, &auxWorkerLive)
}
