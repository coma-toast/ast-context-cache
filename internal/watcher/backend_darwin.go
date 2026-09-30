package watcher

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsevents"
	"github.com/fsnotify/fsnotify"
)

const nativeBackendName = "fsevents"

// fseventsLatency is how long FSEvents coalesces changes before delivering a
// batch. handleFSEvent's per-file debounce sits on top of it.
const fseventsLatency = 200 * time.Millisecond

const (
	fseventsItemChange = fsevents.ItemCreated | fsevents.ItemRemoved | fsevents.ItemRenamed |
		fsevents.ItemModified | fsevents.ItemInodeMetaMod
	fseventsLost = fsevents.MustScanSubDirs | fsevents.KernelDropped | fsevents.UserDropped
)

// fseventsBackend watches a whole tree with one FSEvents stream. kqueue needs
// a descriptor for every watched directory and every file in it; this needs
// none per path, so the cost doesn't grow with repo size or repo count.
type fseventsBackend struct {
	root     string // project path events are reported under
	realRoot string // root with symlinks resolved; FSEvents reports canonical paths
	stream   *fsevents.EventStream
	events   chan fsnotify.Event
	errors   chan error

	closing   chan struct{} // closed first: forwarding stops
	stopped   chan struct{} // closed once the stream is stopped: the loop exits
	closeOnce sync.Once
}

func newNativeBackend(root string) (backend, error) {
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, err
	}
	b := &fseventsBackend{
		root:     filepath.Clean(root),
		realRoot: filepath.Clean(realRoot),
		events:   make(chan fsnotify.Event, 256),
		errors:   make(chan error, 8),
		closing:  make(chan struct{}),
		stopped:  make(chan struct{}),
	}
	b.stream = &fsevents.EventStream{
		Paths:   []string{b.realRoot},
		Latency: fseventsLatency,
		// Not WatchRoot: it holds a directory descriptor open on every
		// ancestor of the root (plus a kqueue) to notice the root moving,
		// ~10 descriptors per project. A moved or deleted project is the
		// purge sweep's job anyway.
		Flags: fsevents.FileEvents | fsevents.NoDefer,
	}
	if err := b.stream.Start(); err != nil {
		return nil, fmt.Errorf("fsevents: %w", err)
	}
	go b.loop()
	return b, nil
}

func (b *fseventsBackend) Add(string) error              { return nil }
func (b *fseventsBackend) Events() <-chan fsnotify.Event { return b.events }
func (b *fseventsBackend) Errors() <-chan error          { return b.errors }
func (b *fseventsBackend) Recursive() bool               { return true }
func (b *fseventsBackend) Name() string                  { return nativeBackendName }
func (b *fseventsBackend) OSWatches() int                { return 1 }

// Close is called with the package mutex held, while the event consumer may be
// blocked waiting for that same mutex, so nothing here may wait on the consumer.
func (b *fseventsBackend) Close() error {
	b.closeOnce.Do(func() {
		// Stop forwarding first, so the loop never blocks on the consumer.
		close(b.closing)
		// The library's callback blocks handing a batch to stream.Events; the
		// loop keeps draining it until Stop returns, so Stop can't wedge on it.
		b.stream.Stop()
		close(b.stopped)
		// A callback already past its registry lookup can still deliver one
		// last batch after Stop; drain briefly so its dispatch thread isn't
		// parked forever.
		go func(ch <-chan []fsevents.Event) {
			t := time.NewTimer(5 * time.Second)
			defer t.Stop()
			for {
				select {
				case <-ch:
				case <-t.C:
					return
				}
			}
		}(b.stream.Events)
	})
	return nil
}

func (b *fseventsBackend) loop() {
	defer close(b.errors)
	defer close(b.events)
	for {
		select {
		case batch := <-b.stream.Events:
			for _, e := range batch {
				if !b.forward(e) {
					break
				}
			}
		case <-b.stopped:
			return
		}
	}
}

// forward translates one FSEvents record and hands it on. It returns false
// once the backend is closing.
func (b *fseventsBackend) forward(e fsevents.Event) bool {
	if e.Flags&fseventsLost != 0 {
		b.sendErr(fsnotify.ErrEventOverflow)
	}
	path, ok := b.projectPath(e.Path)
	if !ok {
		return true
	}
	// A directory moved into or out of the tree arrives as one event for the
	// directory alone, never for the files under it.
	if e.Flags&fsevents.ItemIsDir != 0 && e.Flags&fsevents.ItemRenamed != 0 {
		b.sendErr(fsnotify.ErrEventOverflow)
	}
	if e.Flags&fseventsItemChange == 0 {
		return true
	}
	select {
	case b.events <- fsnotify.Event{Name: path, Op: fseventsOp(path, e.Flags)}:
		return true
	case <-b.closing:
		return false
	}
}

// sendErr never blocks: one queued overflow already triggers a rescan, so a
// full buffer loses nothing that matters.
func (b *fseventsBackend) sendErr(err error) {
	select {
	case b.errors <- err:
	default:
	}
}

// projectPath maps a canonical path from FSEvents back under the project root
// as the watcher keys it. The prefix match ignores case, since APFS is
// case-insensitive by default and FSEvents reports the on-disk spelling.
func (b *fseventsBackend) projectPath(p string) (string, bool) {
	p = filepath.Clean(p)
	if strings.EqualFold(p, b.realRoot) {
		return b.root, true
	}
	n := len(b.realRoot)
	if len(p) > n && p[n] == '/' && strings.EqualFold(p[:n], b.realRoot) {
		return filepath.Join(b.root, p[n+1:]), true
	}
	return "", false
}

// fseventsOp decides the operation from what's on disk now. Flags accumulate
// over the coalescing window (a file can be created, written, and removed in
// one record), so they can't say which state the path ended in.
func fseventsOp(path string, flags fsevents.EventFlags) fsnotify.Op {
	if _, err := os.Lstat(path); errors.Is(err, fs.ErrNotExist) {
		return fsnotify.Remove
	}
	switch {
	case flags&(fsevents.ItemCreated|fsevents.ItemRenamed) != 0:
		return fsnotify.Create
	case flags&fsevents.ItemModified != 0:
		return fsnotify.Write
	}
	return fsnotify.Chmod
}
