package watcher

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
)

// waitForEvent drains b until an event for path satisfies want, or fails.
func waitForEvent(t *testing.T, b backend, path string, want func(fsnotify.Event) bool) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		select {
		case ev, ok := <-b.Events():
			if !ok {
				t.Fatalf("events closed while waiting for %s", path)
			}
			if ev.Name == path && want(ev) {
				return
			}
		case err := <-b.Errors():
			t.Logf("backend error: %v", err)
		case <-deadline:
			t.Fatalf("no matching event for %s", path)
		}
	}
}

func TestFSEventsBackendReportsChangesUnderProjectPath(t *testing.T) {
	// t.TempDir is under /var/folders, a symlink to /private/var, so this also
	// covers mapping FSEvents' canonical paths back to the path as given.
	root := NormalizeProjectPath(t.TempDir())
	b, err := newNativeBackend(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { b.Close() })
	if !b.Recursive() || b.Name() != "fsevents" || b.OSWatches() != 1 {
		t.Fatalf("unexpected backend shape: recursive=%v name=%s watches=%d", b.Recursive(), b.Name(), b.OSWatches())
	}

	// Created after the watch started, deep in a new directory: kqueue would
	// need a fresh Add for it, FSEvents covers it already.
	nested := filepath.Join(root, "a", "b", "c")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(nested, "x.go")
	if err := os.WriteFile(file, []byte("package c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	waitForEvent(t, b, file, func(ev fsnotify.Event) bool { return !ev.Has(fsnotify.Remove) })

	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	waitForEvent(t, b, file, func(ev fsnotify.Event) bool { return ev.Has(fsnotify.Remove) })
}

func TestFSEventsBackendCloseEndsStreams(t *testing.T) {
	root := NormalizeProjectPath(t.TempDir())
	b, err := newNativeBackend(root)
	if err != nil {
		t.Fatal(err)
	}
	// Leave events unread so the translate loop is likely blocked sending when
	// Close runs: Close must not wait on the consumer.
	for i := 0; i < 50; i++ {
		os.WriteFile(filepath.Join(root, "f"+string(rune('a'+i%26))+".go"), []byte("package x\n"), 0o644)
	}
	time.Sleep(500 * time.Millisecond)
	done := make(chan struct{})
	go func() {
		b.Close()
		b.Close() // idempotent
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Close blocked")
	}
	deadline := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-b.Events():
			if !ok {
				return
			}
		case <-deadline:
			t.Fatal("events channel not closed after Close")
		}
	}
}

func TestFSEventsProjectPathMapping(t *testing.T) {
	b := &fseventsBackend{root: "/var/folders/x/proj", realRoot: "/private/var/folders/x/proj"}
	cases := map[string]string{
		"/private/var/folders/x/proj":          "/var/folders/x/proj",
		"/private/var/folders/x/proj/a/b.go":   "/var/folders/x/proj/a/b.go",
		"/private/var/folders/x/PROJ/a/b.go":   "/var/folders/x/proj/a/b.go", // case-insensitive APFS
		"/private/var/folders/x/project2/b.go": "",                           // sibling sharing a prefix
		"/elsewhere/b.go":                      "",
	}
	for in, want := range cases {
		got, ok := b.projectPath(in)
		if want == "" {
			if ok {
				t.Errorf("projectPath(%q) = %q, want no match", in, got)
			}
			continue
		}
		if !ok || got != want {
			t.Errorf("projectPath(%q) = %q, %v; want %q", in, got, ok, want)
		}
	}
}
