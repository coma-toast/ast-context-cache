package watcher

import (
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/fsnotify/fsnotify"
)

// handleFSEvent can't stop a debounce timer that already fired, so an edit
// that arrives while its callback is re-indexing queues a second timer under
// the same key. The first callback used to delete that key when it finished,
// dropping the second timer from debounceTimers: it still fired, but
// DeleteWatcher could no longer find it to cancel, so a file of a project
// deleted before it fired was re-indexed anyway.
func TestFinishedDebounceCallbackLeavesNewerTimerCancellable(t *testing.T) {
	dir := NormalizeProjectPath(t.TempDir())
	path := filepath.Join(dir, "a.go")
	key := debounceKey(dir, path)
	ev := fsnotify.Event{Name: path, Op: fsnotify.Write}
	if err := os.WriteFile(path, []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// The first callback's re-index says it started, waits for finish, and
	// says it returned; any later one waits for the test to end. This used to
	// hold the callback in os.ReadFile on a FIFO, but macOS now and then never
	// wakes a FIFO's blocked reader when the last writer closes, which left
	// cleanupWatchers waiting on that callback forever.
	started, finish, returned, done := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	prev := indexFile
	indexFile = func(string, string) (int, int, int, error) {
		if calls.Add(1) > 1 {
			<-done
			return 0, 0, 0, errs.New("re-index held by test")
		}
		defer close(returned)
		close(started)
		<-finish
		return 0, 0, 0, errs.New("re-index held by test")
	}
	t.Cleanup(func() { indexFile = prev })
	cleanupWatchers(t)
	var finishOnce sync.Once
	finishFirst := func() { finishOnce.Do(func() { close(finish) }) }
	t.Cleanup(func() { // runs before cleanupWatchers waits on the callbacks
		finishFirst()
		close(done)
	})

	handleFSEvent(ev, dir, nil)
	waitFor(t, started, "the first debounce callback never started re-indexing")

	// The file changes again while the first callback is re-indexing it.
	handleFSEvent(ev, dir, nil)
	debounceMu.Lock()
	second := debounceTimers[key]
	debounceMu.Unlock()
	if second == nil {
		t.Fatal("handleFSEvent queued no timer for the second event")
	}

	// Let the first callback finish. The second timer fires 500ms after it
	// was queued, so the checks below normally run well inside that window;
	// if it fires sooner anyway, its callback waits on done and leaves its key
	// alone.
	finishFirst()
	waitFor(t, returned, "the first debounce callback never returned from re-indexing")
	time.Sleep(100 * time.Millisecond) // for the callback's bookkeeping after the re-index

	debounceMu.Lock()
	got := debounceTimers[key]
	debounceMu.Unlock()
	if got != second {
		t.Error("the finished callback removed the newer timer from debounceTimers")
	}
	DeleteWatcher(dir)
	if second.Stop() {
		bg.Done() // as stopDebounce does, for the callback that now won't run
		t.Error("DeleteWatcher left the newer timer pending, so it would re-index a file of the deleted project")
	}
}

// waitFor fails t if ch isn't closed within 10s, so a callback that never
// gets where the test expects can't hang the package.
func waitFor(t *testing.T, ch <-chan struct{}, msg string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		t.Fatal(msg)
	}
}
