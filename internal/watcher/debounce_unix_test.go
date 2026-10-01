//go:build darwin || linux

package watcher

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

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
	cleanupWatchers(t)
	path := filepath.Join(dir, "a.go")
	key := debounceKey(dir, path)
	ev := fsnotify.Event{Name: path, Op: fsnotify.Write}

	// The first callback re-indexes a FIFO. IndexFile's os.ReadFile blocks
	// opening it until the test opens the write end, and reading it until the
	// test closes that end, so the test knows when the callback is in flight
	// and decides when it finishes.
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Fatal(err)
	}
	handleFSEvent(ev, dir, nil)
	debounceMu.Lock()
	_, queued := debounceTimers[key]
	debounceMu.Unlock()
	if !queued {
		t.Fatal("handleFSEvent queued no debounce timer") // the open below would block forever
	}
	fifo, err := os.OpenFile(path, os.O_WRONLY, 0) // returns once the callback is reading
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { fifo.Close() }) // before cleanupWatchers waits on the callback

	// The file changes again. The callback keeps reading the FIFO it opened.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	handleFSEvent(ev, dir, nil)
	debounceMu.Lock()
	second := debounceTimers[key]
	debounceMu.Unlock()
	if second == nil {
		t.Fatal("handleFSEvent queued no timer for the second event")
	}

	// Let the first callback finish. The second timer fires 500ms after it
	// was queued, so the checks below run well inside that window.
	if _, err := fifo.WriteString("package a\n"); err != nil {
		t.Fatal(err)
	}
	fifo.Close()
	time.Sleep(200 * time.Millisecond)

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
