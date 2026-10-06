package watcher

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/fsnotify/fsnotify"
)

// The dashboard holds a project's newest indexing toast open while Pending
// reports work left, and closes it on QueueDrainedHook, so the hook must fire
// once, only after the last queued re-index finishes.
func TestQueueDrainedHookFiresAfterLastReindex(t *testing.T) {
	dir := NormalizeProjectPath(t.TempDir())
	release := map[string]chan struct{}{}
	started := make(chan string, 2)
	for _, name := range []string{"a.go", "b.go"} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("package a\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		release[path] = make(chan struct{})
	}
	prev := indexFile
	indexFile = func(path, _ string) (int, int, int, error) {
		started <- path
		<-release[path]
		return 0, 0, 0, errs.New("re-index held by test")
	}
	t.Cleanup(func() { indexFile = prev })
	drained := make(chan uint64, 4)
	prevHook := QueueDrainedHook
	QueueDrainedHook = func(projectPath string, gen uint64) {
		if projectPath == dir {
			drained <- gen
		}
	}
	t.Cleanup(func() { QueueDrainedHook = prevHook })
	cleanupWatchers(t)
	t.Cleanup(func() { // runs before cleanupWatchers waits on the callbacks
		for _, ch := range release {
			select {
			case <-ch:
			default:
				close(ch)
			}
		}
	})

	for path := range release {
		handleFSEvent(fsnotify.Event{Name: path, Op: fsnotify.Write}, dir, nil)
	}
	if n, gen := Pending(dir); n != 2 || gen != 0 {
		t.Fatalf("Pending = %d, gen %d before any re-index; want 2, 0", n, gen)
	}
	first := <-started
	second := <-started

	close(release[first])
	time.Sleep(100 * time.Millisecond) // for the callback's bookkeeping after the re-index
	if n, _ := Pending(dir); n != 1 {
		t.Fatalf("Pending = %d with one re-index still running; want 1", n)
	}
	select {
	case gen := <-drained:
		t.Fatalf("QueueDrainedHook fired (gen %d) with a re-index still running", gen)
	default:
	}

	close(release[second])
	select {
	case gen := <-drained:
		if gen != 1 {
			t.Fatalf("drain generation = %d; want 1", gen)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("QueueDrainedHook never fired after the last re-index")
	}
	if n, gen := Pending(dir); n != 0 || gen != 1 {
		t.Fatalf("Pending = %d, gen %d after draining; want 0, 1", n, gen)
	}
}
