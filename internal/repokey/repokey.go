// Package repokey resolves the shared git identity of a checkout so that
// worktrees of the same repository (WTG spaces keep one checkout per space,
// each on its own branch) are recognisable as siblings of one logical repo.
//
// It deliberately depends on nothing but the standard library: indexing,
// scoping and dashboard code all need repo identity, and a leaf package keeps
// those callers free of import cycles.
package repokey

import (
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const cacheTTL = 45 * time.Second

type cacheEntry struct {
	at  time.Time
	key string
}

var (
	mu    sync.Mutex
	cache = map[string]cacheEntry{}
)

// Normalize returns a cleaned absolute project path.
func Normalize(projectPath string) string {
	projectPath = strings.TrimSpace(projectPath)
	if projectPath == "" {
		return ""
	}
	if abs, err := filepath.Abs(projectPath); err == nil {
		projectPath = abs
	}
	return filepath.Clean(projectPath)
}

// Key returns the resolved git common dir for projectPath. Every worktree of a
// repository resolves to the same key. Paths without usable git metadata fall
// back to their own cleaned path so unrelated projects never collide.
func Key(projectPath string) string {
	path := Normalize(projectPath)
	if path == "" {
		return ""
	}
	mu.Lock()
	e, ok := cache[path]
	mu.Unlock()
	if ok && time.Since(e.at) < cacheTTL {
		return e.key
	}
	key := resolve(path)
	mu.Lock()
	cache[path] = cacheEntry{at: time.Now(), key: key}
	mu.Unlock()
	return key
}

// SameRepo reports whether two project paths are checkouts of the same repo.
func SameRepo(a, b string) bool {
	ka := Key(a)
	if ka == "" {
		return false
	}
	return ka == Key(b)
}

// Invalidate clears the cached key for a path (or all keys when path is empty).
func Invalidate(projectPath string) {
	path := Normalize(projectPath)
	mu.Lock()
	defer mu.Unlock()
	if path == "" {
		cache = map[string]cacheEntry{}
		return
	}
	delete(cache, path)
}

func resolve(path string) string {
	out, err := exec.Command("git", "-C", path, "rev-parse", "--git-common-dir").Output()
	if err != nil {
		return path
	}
	common := strings.TrimSpace(string(out))
	if common == "" {
		return path
	}
	// git reports an absolute common dir for linked worktrees and a relative
	// ".git" for a plain checkout; only the relative form joins onto path.
	if !filepath.IsAbs(common) {
		common = filepath.Join(path, common)
	}
	if resolved, err := filepath.EvalSymlinks(common); err == nil {
		common = resolved
	}
	return filepath.Clean(common)
}
