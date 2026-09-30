package watcher

import (
	"log"
	"os"
	"runtime"
	"strings"

	"github.com/fsnotify/fsnotify"
)

// watcherBackendEnv set to "fsnotify" forces the portable backend on every OS
// (e.g. to compare fd usage, or for a volume FSEvents doesn't report on).
const watcherBackendEnv = "AST_MCP_WATCHER_BACKEND"

// backend is the OS-level watch behind one project watcher. Every backend
// delivers fsnotify-shaped events, so handleFSEvent has a single input.
type backend interface {
	// Add watches one more directory. Recursive backends already cover the
	// whole tree under the root and ignore it.
	Add(dir string) error
	Events() <-chan fsnotify.Event
	// Errors carries backend errors. fsnotify.ErrEventOverflow means events
	// were lost and the tree needs a catch-up rescan.
	Errors() <-chan error
	Close() error
	// Recursive reports whether watching the root covers every subdirectory,
	// so StartWatcher can skip the per-directory Add walk.
	Recursive() bool
	// Name identifies the mechanism in status output.
	Name() string
	// OSWatches is how many kernel watches the backend holds: one per watched
	// directory for fsnotify (under kqueue each also costs a descriptor per
	// file in it), one per root for FSEvents.
	OSWatches() int
}

// newBackend opens the watch for one project root: the OS's native recursive
// backend when there is one (FSEvents on macOS), else fsnotify.
func newBackend(root string) (backend, error) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(watcherBackendEnv)), "fsnotify") {
		return newFsnotifyBackend()
	}
	b, err := newNativeBackend(root)
	if err != nil {
		log.Printf("Watcher: native backend unavailable for %s (%v); falling back to %s", root, err, fsnotifyBackendName())
	}
	if b != nil {
		return b, nil
	}
	return newFsnotifyBackend()
}

// DefaultBackendName is the backend new watchers get on this machine.
func DefaultBackendName() string {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(watcherBackendEnv)), "fsnotify") {
		return fsnotifyBackendName()
	}
	if nativeBackendName != "" {
		return nativeBackendName
	}
	return fsnotifyBackendName()
}

type fsnotifyBackend struct{ w *fsnotify.Watcher }

func newFsnotifyBackend() (backend, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	return &fsnotifyBackend{w: w}, nil
}

func (b *fsnotifyBackend) Add(dir string) error          { return b.w.Add(dir) }
func (b *fsnotifyBackend) Events() <-chan fsnotify.Event { return b.w.Events }
func (b *fsnotifyBackend) Errors() <-chan error          { return b.w.Errors }
func (b *fsnotifyBackend) Close() error                  { return b.w.Close() }
func (b *fsnotifyBackend) Recursive() bool               { return false }
func (b *fsnotifyBackend) Name() string                  { return fsnotifyBackendName() }
func (b *fsnotifyBackend) OSWatches() int                { return len(b.w.WatchList()) }

func fsnotifyBackendName() string {
	switch runtime.GOOS {
	case "linux":
		return "inotify"
	case "windows":
		return "ReadDirectoryChangesW"
	case "darwin", "freebsd", "openbsd", "netbsd", "dragonfly":
		return "kqueue"
	}
	return "fsnotify"
}
