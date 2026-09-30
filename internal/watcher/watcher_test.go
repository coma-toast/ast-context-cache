package watcher

import (
	"path/filepath"
	"testing"
	"time"
)

func TestNormalizeProjectPath(t *testing.T) {
	dir := t.TempDir()
	got := NormalizeProjectPath(dir)
	if got == "" {
		t.Fatal("expected non-empty path")
	}
	if got != NormalizeProjectPath(dir+"/") {
		t.Fatalf("trailing slash mismatch: %q vs %q", got, NormalizeProjectPath(dir+"/"))
	}
}

// A debounce timer queued by handleFSEvent just before a project is deleted
// must not fire afterward and re-index (or delete symbols for) a file the
// delete just purged — with no watcher left to have caused it.
func TestDeleteWatcherCancelsPendingDebounceTimersForItsProject(t *testing.T) {
	debounceMu.Lock()
	debounceTimers = map[string]*time.Timer{}
	debounceMu.Unlock()

	deletedProject := NormalizeProjectPath(t.TempDir())
	otherProject := NormalizeProjectPath(t.TempDir())

	var deletedFired, otherFired bool
	debounceMu.Lock()
	debounceTimers[filepath.Join(deletedProject, "a.go")] = time.AfterFunc(50*time.Millisecond, func() { deletedFired = true })
	debounceTimers[filepath.Join(otherProject, "b.go")] = time.AfterFunc(50*time.Millisecond, func() { otherFired = true })
	debounceMu.Unlock()

	DeleteWatcher(deletedProject)

	time.Sleep(100 * time.Millisecond)
	if deletedFired {
		t.Fatal("debounce timer for the deleted project should have been cancelled, not fired")
	}
	if !otherFired {
		t.Fatal("debounce timer for an unrelated project should not be cancelled")
	}

	debounceMu.Lock()
	_, stillTracked := debounceTimers[filepath.Join(deletedProject, "a.go")]
	debounceMu.Unlock()
	if stillTracked {
		t.Fatal("cancelled timer should be removed from debounceTimers")
	}
}

func TestDebounceKeyOwnership(t *testing.T) {
	space := "/s/space"
	repo := "/s/space/repo"
	cases := []struct {
		key     string
		project string
		want    bool
	}{
		{debounceKey(repo, repo+"/a.go"), repo, true},
		{debounceKey(repo, repo+"/a.go"), space, false}, // a nested project's timer
		{debounceKey(space, repo+"/a.go"), space, true},
		{debounceKey(space, repo+"/a.go"), repo, false},
		{space + "/\x00catch-up", space, true},
		{repo + "/\x00catch-up", space, false},
		{repo + "/a.go", space, true}, // bare path keys keep prefix matching
		{"/s/space-other/a.go", space, false},
	}
	for _, c := range cases {
		if got := debounceKeyOwnedBy(c.key, c.project); got != c.want {
			t.Errorf("debounceKeyOwnedBy(%q, %q) = %v, want %v", c.key, c.project, got, c.want)
		}
	}
}
