package watcher

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// idleLoop starts in init() and used to read settings on every tick, even
// with no watcher running to stop. A tick after TestMain's db.Close raced
// with it. This one only fails under -race.
func TestIdleTickLeavesDBAloneWithNoActiveWatcher(t *testing.T) {
	stopWatchersForTest() // nothing of ours may be using the pools when they close
	t.Cleanup(func() {
		if err := db.Init(); err != nil { // reopen for the tests after this one
			t.Fatal(err)
		}
	})

	// The tick lands just after Close, as it does at exit. Run first instead,
	// its query takes the pool's lock before Close does, which orders the two
	// and hides the race.
	done := make(chan struct{})
	go func() {
		defer close(done)
		time.Sleep(20 * time.Millisecond)
		idleTick()
	}()
	db.Close()
	<-done
}
