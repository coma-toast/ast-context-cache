package watcher

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

// ContainerRootsFunc lists directories that hold many separate projects (the
// wtg spaces root, the repo discovery root). main sets it to
// projectmeta.ContainerRoots; watcher can't import projectmeta, which imports
// watcher.
var ContainerRootsFunc func() []string

var (
	refusalMu     sync.Mutex
	refusalLogged = map[string]bool{}
)

// WatchRefusal returns why projectPath must not get a watcher, or "" if it may.
//
// A watcher on a directory of projects (~/spaces, ~/git, $HOME, or any of
// their ancestors) walks and re-indexes every project below it under its own
// key and never goes idle, since some project under it always changes. An MCP
// client started in such a directory passes it as project_path, so this is
// checked on every start rather than trusted to not happen.
func WatchRefusal(projectPath string) string {
	return watchRefusalFor(projectPath, containerRoots())
}

// containerRoots is $HOME plus ContainerRootsFunc's roots, normalized. It
// reads the wtg config, so callers checking many projects fetch it once.
func containerRoots() []string {
	var raw []string
	if home, err := os.UserHomeDir(); err == nil {
		raw = append(raw, home)
	}
	if ContainerRootsFunc != nil {
		raw = append(raw, ContainerRootsFunc()...)
	}
	roots := make([]string, 0, len(raw))
	for _, r := range raw {
		if r = NormalizeProjectPath(r); r != "" {
			roots = append(roots, r)
		}
	}
	return roots
}

func watchRefusalFor(projectPath string, roots []string) string {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return ""
	}
	if projectPath == string(filepath.Separator) {
		return "the filesystem root holds every project on the machine"
	}
	for _, r := range roots {
		if projectPath == r {
			return fmt.Sprintf("%s holds many separate projects; watchers run per repo inside it", projectPath)
		}
		if strings.HasPrefix(r, projectPath+string(filepath.Separator)) {
			return fmt.Sprintf("%s contains %s, which holds many separate projects; watchers run per repo inside it", projectPath, r)
		}
	}
	return ""
}

func logRefusalOnce(projectPath, reason string) {
	refusalMu.Lock()
	seen := refusalLogged[projectPath]
	refusalLogged[projectPath] = true
	refusalMu.Unlock()
	if !seen {
		log.Printf("Watcher not started for %s: %s", projectPath, reason)
	}
}

// underSkippedDir reports whether path sits below a directory the watcher
// never descends into (indexer.ShouldSkipDir: dot-dirs, node_modules, ...).
// The walk-based backend never watches those, so it never sees their events;
// a recursive backend reports the whole tree and needs this filter instead.
func underSkippedDir(path, projectPath string) bool {
	rel, err := filepath.Rel(projectPath, path)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	for dir := filepath.Dir(rel); dir != "." && dir != string(filepath.Separator); dir = filepath.Dir(dir) {
		if indexer.ShouldSkipDir(filepath.Base(dir)) {
			return true
		}
	}
	return false
}
