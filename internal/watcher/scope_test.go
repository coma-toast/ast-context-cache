package watcher

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
)

func TestWatchRefusalBlocksContainerRootsAndTheirAncestors(t *testing.T) {
	base := t.TempDir()
	home := filepath.Join(base, "home")
	spaces := filepath.Join(home, "spaces")
	git := filepath.Join(home, "git")
	for _, d := range []string{spaces, git} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("HOME", home)
	prev := ContainerRootsFunc
	ContainerRootsFunc = func() []string { return []string{spaces, git} }
	t.Cleanup(func() { ContainerRootsFunc = prev })

	blocked := []string{spaces, spaces + "/", git, home, base, "/"}
	for _, p := range blocked {
		if WatchRefusal(p) == "" {
			t.Errorf("WatchRefusal(%q) = \"\", want a refusal", p)
		}
	}
	allowed := []string{
		filepath.Join(spaces, "ai-foo"),             // a space root: bounded by that space
		filepath.Join(spaces, "ai-foo", "slapi"),    // a worktree
		filepath.Join(git, "sandbox"),               // a clone
		filepath.Join(home, "configSync"),           // anything else under $HOME
		filepath.Join(home, "spaces-archive"),       // prefix of a root's name, not an ancestor
		filepath.Join(base, "elsewhere", "project"), // unrelated
	}
	for _, p := range allowed {
		if r := WatchRefusal(p); r != "" {
			t.Errorf("WatchRefusal(%q) = %q, want allowed", p, r)
		}
	}
}

func TestStartWatcherRefusesContainerRoot(t *testing.T) {
	container := NormalizeProjectPath(t.TempDir())
	prev := ContainerRootsFunc
	ContainerRootsFunc = func() []string { return []string{container} }
	t.Cleanup(func() { ContainerRootsFunc = prev })

	EnsureWatcher(container)
	t.Cleanup(func() { DeleteWatcher(container) })
	if IsActive(container) {
		t.Fatal("a container root must not get a watcher")
	}
	st := ProjectStatus(container)
	if reason, _ := st["blocked_reason"].(string); !strings.Contains(reason, "separate projects") {
		t.Fatalf("ProjectStatus blocked_reason = %q, want the container refusal", reason)
	}
}

func TestUnderSkippedDir(t *testing.T) {
	root := "/work/repo"
	cases := map[string]bool{
		"/work/repo/main.go":                        false,
		"/work/repo/internal/watcher/watcher.go":    false,
		"/work/repo/.git/objects/ab/cdef":           true,
		"/work/repo/.git/index.lock":                true,
		"/work/repo/ui/node_modules/react/index.js": true,
		"/work/repo/pkg/.cache/x.go":                true,
		"/work/repo/.gitignore":                     false, // a dotfile itself, not under a dot-dir
		"/work/repo":                                false,
		"/work/other/x.go":                          false, // outside: left to the existing checks
	}
	for path, want := range cases {
		if got := underSkippedDir(path, root); got != want {
			t.Errorf("underSkippedDir(%q) = %v, want %v", path, got, want)
		}
	}
}

// With a recursive backend, .git churn (every git command) reaches
// handleFSEvent. It must not count as activity, or no watcher would ever idle.
func TestHandleFSEventIgnoresSkippedDirsForActivity(t *testing.T) {
	project := NormalizeProjectPath(t.TempDir())
	if err := os.MkdirAll(filepath.Join(project, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	stale := time.Now().Add(-time.Hour)
	mu.Lock()
	lastActivity[project] = stale
	mu.Unlock()
	t.Cleanup(func() { DeleteWatcher(project) })

	handleFSEvent(fsnotify.Event{Name: filepath.Join(project, ".git", "index.lock"), Op: fsnotify.Create}, project, nil)
	mu.Lock()
	got := lastActivity[project]
	mu.Unlock()
	if !got.Equal(stale) {
		t.Fatalf("a .git event bumped lastActivity to %v", got)
	}
}
