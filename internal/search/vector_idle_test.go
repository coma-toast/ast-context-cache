package search

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// The idle loop started in init() used to read db's pools on every tick,
// even with nothing loaded to unload. A test closing the pools meanwhile (as
// the watcher package's TestMain does before exiting) raced with it. This
// one only fails under -race.
func TestIdleTickLeavesDBAloneWhenNothingIsLoaded(t *testing.T) {
	prev := db.SetHomeForTest(t.TempDir())
	defer prev()
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	// The tick lands just after Close, as it did in the watcher package. Run
	// first instead, its query takes the pool's lock before Close does, which
	// orders the two and hides the race.
	vc := &VectorCache{stopIdle: make(chan struct{})}
	done := make(chan struct{})
	go func() {
		defer close(done)
		time.Sleep(20 * time.Millisecond)
		vc.idleTick()
	}()
	db.Close()
	<-done
}

func TestIdleTickUnloadsCacheIdlePastTimeout(t *testing.T) {
	prev := db.SetHomeForTest(t.TempDir())
	defer prev()
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	vc := &VectorCache{
		entries:  []VectorEntry{{ID: 1}},
		loaded:   true,
		lastUsed: time.Now().Add(-time.Hour), // idle_unload_minutes defaults to 1
		stopIdle: make(chan struct{}),
	}
	vc.idleTick()
	if vc.loaded || vc.entries != nil {
		t.Fatal("a cache idle past the timeout should be unloaded")
	}
}
