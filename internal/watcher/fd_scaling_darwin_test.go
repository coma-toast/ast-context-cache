package watcher

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

const (
	scaleDirsPerProject = 10
	scaleFilesPerDir    = 10
	scaleFilesPerProj   = scaleDirsPerProject * scaleFilesPerDir
)

// makeScaleProject writes a small Go tree: kqueue would hold a descriptor for
// every one of its directories and files.
func makeScaleProject(t *testing.T, root string) {
	t.Helper()
	for d := 0; d < scaleDirsPerProject; d++ {
		dir := filepath.Join(root, fmt.Sprintf("pkg%d", d))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for f := 0; f < scaleFilesPerDir; f++ {
			src := fmt.Sprintf("package pkg%d\n\nfunc F%d() int { return %d }\n", d, f, f)
			if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%d.go", f)), []byte(src), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func waitIndexed(t *testing.T, project string, want func(map[string]time.Time) bool, what string) {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		if want(db.GetIndexedFiles(project)) {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("%s: timed out (project %s has %d indexed files)", what, project, len(db.GetIndexedFiles(project)))
}

func openFDs(t *testing.T) int {
	t.Helper()
	u := sys.FileDescriptorUsage()
	if !u.Available {
		t.Fatal("fd count unavailable")
	}
	return u.Open
}

// The 2026-09-30 failure: kqueue held one descriptor per watched file and
// directory, so the shared server hit kern.maxfilesperproc (61,440) with ~90
// wtg worktrees watched. On macOS each watcher must now cost O(1) descriptors
// whatever the tree size, and still pick up edits.
func TestWatchersOnManyProjectsStayCheapInFileDescriptors(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv(watcherBackendEnv, "")
	if err := db.Init(); err != nil {
		t.Fatalf("db init: %v", err)
	}
	base := t.TempDir()

	const projects = 12
	var paths []string
	for i := 0; i < projects; i++ {
		p := NormalizeProjectPath(filepath.Join(base, fmt.Sprintf("repo%02d", i)))
		makeScaleProject(t, p)
		paths = append(paths, p)
	}
	// Open the DB pools first, so their connections don't count as watcher cost.
	db.GetIndexedFiles(paths[0])
	before := openFDs(t)

	for _, p := range paths {
		StartWatcher(p)
		p := p
		t.Cleanup(func() { DeleteWatcher(p) })
	}
	for _, p := range paths {
		waitIndexed(t, p, func(m map[string]time.Time) bool { return len(m) >= scaleFilesPerProj }, "catch-up")
	}
	delta := openFDs(t) - before
	kqueueCost := projects * (scaleFilesPerProj + scaleDirsPerProject + 1)
	t.Logf("%d watchers over %d files: +%d descriptors (kqueue would hold ~%d)", projects, projects*scaleFilesPerProj, delta, kqueueCost)
	// A stream itself costs no descriptors; the headroom is for SQLite
	// connections the catch-up writes opened. Tight enough to also catch the
	// WatchRoot flag coming back (a descriptor per ancestor directory of each
	// root, ~13 per project here).
	if limit := 2*projects + 40; delta > limit {
		t.Fatalf("watchers cost %d descriptors, want <= %d", delta, limit)
	}
	for _, p := range paths {
		if st := ProjectStatus(p); st["backend"] != "fsevents" {
			t.Fatalf("%s: backend %v, want fsevents", p, st["backend"])
		}
	}

	// Edits are still picked up: a new file deep in a new directory, then its removal.
	target := paths[projects/2]
	added := filepath.Join(target, "added", "deeper", "new.go")
	if err := os.MkdirAll(filepath.Dir(added), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(added, []byte("package deeper\n\nfunc Added() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	waitIndexed(t, target, func(m map[string]time.Time) bool { _, ok := m[added]; return ok }, "new file indexed")
	if err := os.Remove(added); err != nil {
		t.Fatal(err)
	}
	waitIndexed(t, target, func(m map[string]time.Time) bool { _, ok := m[added]; return !ok }, "removed file purged")
}

// Control for the test above: the same tree under the kqueue backend costs a
// descriptor per path, so a regression back to it would fail that test.
func TestFsnotifyBackendCostsDescriptorPerPath(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv(watcherBackendEnv, "fsnotify")
	if err := db.Init(); err != nil {
		t.Fatalf("db init: %v", err)
	}
	p := NormalizeProjectPath(filepath.Join(t.TempDir(), "repo"))
	makeScaleProject(t, p)
	db.GetIndexedFiles(p)
	before := openFDs(t)
	StartWatcher(p)
	t.Cleanup(func() { DeleteWatcher(p) })
	if st := ProjectStatus(p); st["backend"] != "kqueue" {
		t.Fatalf("backend %v, want kqueue", st["backend"])
	}
	delta := openFDs(t) - before
	t.Logf("one kqueue watcher over %d files: +%d descriptors", scaleFilesPerProj, delta)
	if delta < scaleFilesPerProj {
		t.Fatalf("kqueue watcher cost only %d descriptors; the fd measurement isn't seeing per-path cost", delta)
	}
}

// FSEvents reports a directory moved into the tree as one event for the
// directory, none for the files in it. That triggers a catch-up rescan, which
// indexes them.
func TestDirectoryMovedIntoProjectIsIndexed(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv(watcherBackendEnv, "")
	if err := db.Init(); err != nil {
		t.Fatalf("db init: %v", err)
	}
	prev := catchUpAfterLostEvents
	catchUpAfterLostEvents = 100 * time.Millisecond
	t.Cleanup(func() { catchUpAfterLostEvents = prev })

	base := t.TempDir()
	p := NormalizeProjectPath(filepath.Join(base, "repo"))
	if err := os.MkdirAll(p, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(base, "staging", "moved")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "m.go"), []byte("package moved\n\nfunc Moved() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	StartWatcher(p)
	t.Cleanup(func() { DeleteWatcher(p) })
	time.Sleep(300 * time.Millisecond) // let the stream settle before the move

	if err := os.Rename(outside, filepath.Join(p, "moved")); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(p, "moved", "m.go")
	waitIndexed(t, p, func(m map[string]time.Time) bool { _, ok := m[want]; return ok }, "moved-in file indexed")
}

// A space root and a repo inside it are both watched when sessions start in
// each, so both see every change under the repo. Pending re-indexes were keyed
// by file path alone, so the second project's event replaced the first's
// timer and only one project ever saw the change (found validating this
// branch against ~/spaces: a deleted file stayed indexed under the space root).
func TestNestedProjectsBothPickUpChanges(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv(watcherBackendEnv, "")
	if err := db.Init(); err != nil {
		t.Fatalf("db init: %v", err)
	}
	outer := NormalizeProjectPath(filepath.Join(t.TempDir(), "space"))
	inner := filepath.Join(outer, "repo")
	if err := os.MkdirAll(inner, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{outer, inner} {
		StartWatcher(p)
		p := p
		t.Cleanup(func() { DeleteWatcher(p) })
	}
	time.Sleep(300 * time.Millisecond)

	file := filepath.Join(inner, "shared.go")
	if err := os.WriteFile(file, []byte("package repo\n\nfunc Shared() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{outer, inner} {
		waitIndexed(t, p, func(m map[string]time.Time) bool { _, ok := m[file]; return ok }, "indexed under "+p)
	}
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{outer, inner} {
		waitIndexed(t, p, func(m map[string]time.Time) bool { _, ok := m[file]; return !ok }, "purged under "+p)
	}
}
