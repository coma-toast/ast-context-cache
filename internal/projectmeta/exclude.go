package projectmeta

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const excludeSettingKey = "project_exclude_paths"

var (
	excludeCacheRaw    string
	excludeCacheParsed []string
	excludeCacheMu     sync.Mutex
)

// ExcludeList returns configured project exclusion patterns (DB + WTG config).
func ExcludeList() []string {
	excludeCacheMu.Lock()
	defer excludeCacheMu.Unlock()
	raw := strings.TrimSpace(db.GetSetting(excludeSettingKey, ""))
	if raw == excludeCacheRaw && excludeCacheParsed != nil {
		return append([]string(nil), excludeCacheParsed...)
	}
	out := parseExcludeRaw(raw)
	for _, p := range loadWTGConfig().Discovery.Exclude {
		p = strings.TrimSpace(p)
		if p != "" {
			out = appendUnique(out, expandHome(p))
		}
	}
	excludeCacheRaw = raw
	excludeCacheParsed = out
	return append([]string(nil), out...)
}

// InvalidateExcludeCache clears cached exclusion patterns after settings change.
func InvalidateExcludeCache() {
	excludeCacheMu.Lock()
	excludeCacheRaw = ""
	excludeCacheParsed = nil
	excludeCacheMu.Unlock()
}

// ExcludeJSONForSettings returns stored JSON or an empty array for the settings textarea.
func ExcludeJSONForSettings(stored string) string {
	stored = strings.TrimSpace(stored)
	if stored != "" && stored != "[]" {
		var p []string
		if err := json.Unmarshal([]byte(stored), &p); err == nil {
			if b, err := json.MarshalIndent(p, "", "  "); err == nil {
				return string(b)
			}
			return stored
		}
	}
	return "[]"
}

// IsExcluded reports whether a repo path should not be auto-discovered or listed as a project.
func IsExcluded(projectPath string) bool {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return false
	}
	return matchProjectExclude(projectPath, ExcludeList())
}

func matchProjectExclude(projectPath string, patterns []string) bool {
	if len(patterns) == 0 {
		return false
	}
	path := filepath.ToSlash(projectPath)
	var globs []string
	for _, pattern := range patterns {
		pattern = strings.TrimSpace(pattern)
		if pattern == "" {
			continue
		}
		pattern = filepath.ToSlash(expandHome(pattern))
		if strings.HasPrefix(pattern, "basename:") {
			if filepath.Base(projectPath) == strings.TrimPrefix(pattern, "basename:") {
				return true
			}
			continue
		}
		if !strings.ContainsAny(pattern, "*?[") {
			if path == pattern || strings.HasPrefix(path, strings.TrimSuffix(pattern, "/")+"/") {
				return true
			}
			continue
		}
		globs = append(globs, pattern)
	}
	return ignorepatterns.MatchAnyPath(path, globs)
}

func parseExcludeRaw(raw string) []string {
	if raw == "" || raw == "[]" {
		return nil
	}
	var out []string
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil
	}
	var cleaned []string
	for _, p := range out {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		cleaned = append(cleaned, expandHome(p))
	}
	return cleaned
}

func appendUnique(dst []string, v string) []string {
	for _, d := range dst {
		if d == v {
			return dst
		}
	}
	return append(dst, v)
}
