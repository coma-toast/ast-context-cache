package watcher

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

var (
	ignoreCacheRaw    string
	ignoreCacheParsed []string
	ignoreMu          sync.Mutex
)

// GetWatcherIgnorePatterns returns glob patterns from settings key watcher_ignore_globs (JSON array of strings).
func GetWatcherIgnorePatterns() []string {
	ignoreMu.Lock()
	defer ignoreMu.Unlock()
	raw := db.GetSetting("watcher_ignore_globs", "[]")
	if raw == ignoreCacheRaw && ignoreCacheParsed != nil {
		return ignoreCacheParsed
	}
	var out []string
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		out = nil
	}
	for i := range out {
		out[i] = strings.TrimSpace(out[i])
	}
	ignoreCacheRaw = raw
	ignoreCacheParsed = out
	return ignoreCacheParsed
}

// MatchWatcherIgnore reports whether absPath (file) matches any pattern relative to projectPath.
// Patterns use forward slashes; filepath.Match rules apply where * and ? are used.
// A pattern without glob characters matches exact path or any path under that prefix (dir).
// **/suffix matches rel ending with suffix; prefix/** matches under prefix.
func MatchWatcherIgnore(absPath, projectPath string, patterns []string) bool {
	if len(patterns) == 0 {
		return false
	}
	rel, err := filepath.Rel(projectPath, absPath)
	if err != nil {
		return false
	}
	rel = filepath.ToSlash(rel)
	if strings.HasPrefix(rel, "../") {
		return false
	}
	for _, p := range patterns {
		if p == "" {
			continue
		}
		if matchOne(rel, filepath.ToSlash(p)) {
			return true
		}
	}
	return false
}

func matchOne(rel, pattern string) bool {
	if pattern == "" {
		return false
	}
	// **/rest — match anywhere in path
	if strings.HasPrefix(pattern, "**/") {
		suf := strings.TrimPrefix(pattern, "**/")
		if suf == "" {
			return true
		}
		if rel == suf {
			return true
		}
		if strings.HasSuffix(rel, "/"+suf) {
			return true
		}
		return strings.Contains(rel, "/"+suf+"/")
	}
	// pre/** — directory tree
	if strings.HasSuffix(pattern, "/**") {
		pre := strings.TrimSuffix(pattern, "/**")
		if pre == "" {
			return true
		}
		return rel == pre || strings.HasPrefix(rel, pre+"/")
	}
	// No glob: exact path or directory prefix
	if !strings.ContainsAny(pattern, "*?[") {
		return rel == pattern || strings.HasPrefix(rel, pattern+"/")
	}
	if ok, _ := filepath.Match(pattern, rel); ok {
		return true
	}
	if ok, _ := filepath.Match(pattern, filepath.Base(rel)); ok {
		return true
	}
	parts := strings.Split(rel, "/")
	for i := range parts {
		sub := strings.Join(parts[i:], "/")
		if ok, _ := filepath.Match(pattern, sub); ok {
			return true
		}
	}
	return false
}
