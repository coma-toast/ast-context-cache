package ignorepatterns

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const settingKey = "watcher_ignore_globs"

// DefaultGlobs are applied when watcher_ignore_globs is unset or [].
// Users can edit or remove entries in dashboard Settings.
var DefaultGlobs = []string{
	"**/node_modules/**",
	"**/vendor/**",
	"**/bower_components/**",
	"**/Pods/**",
	"**/third_party/**",
	"**/__pycache__/**",
	"**/site-packages/**",
	"**/venv/**",
	"**/.venv/**",
	"**/env/**",
	"**/dist/**",
	"**/build/**",
	"**/out/**",
	"**/target/**",
	"**/coverage/**",
	"**/tmp/**",
	"**/.astcache/**",
	"**/*.pb.go",
	"**/*.min.js",
	"**/*.min.css",
}

var (
	cacheRaw    string
	cacheParsed []string
	cacheMu     sync.Mutex
)

// EnsureDefaults seeds watcher_ignore_globs when missing or empty.
func EnsureDefaults() {
	raw := strings.TrimSpace(db.GetSetting(settingKey, ""))
	if raw != "" && raw != "[]" {
		return
	}
	_ = db.SetSetting(settingKey, JSON(DefaultGlobs))
	InvalidateCache()
}

// List returns configured patterns, falling back to DefaultGlobs when unset or [].
func List() []string {
	cacheMu.Lock()
	defer cacheMu.Unlock()
	raw := strings.TrimSpace(db.GetSetting(settingKey, ""))
	if raw == cacheRaw && cacheParsed != nil {
		return append([]string(nil), cacheParsed...)
	}
	out := parseRaw(raw)
	if len(out) == 0 {
		out = append([]string(nil), DefaultGlobs...)
	}
	cacheRaw = raw
	cacheParsed = out
	return append([]string(nil), out...)
}

// JSONForSettings returns the stored JSON or pretty-printed defaults for the settings textarea.
func JSONForSettings(stored string) string {
	stored = strings.TrimSpace(stored)
	if stored != "" && stored != "[]" {
		var p []string
		if err := json.Unmarshal([]byte(stored), &p); err == nil && len(p) > 0 {
			if b, err := json.MarshalIndent(p, "", "  "); err == nil {
				return string(b)
			}
			return stored
		}
	}
	return JSONPretty(DefaultGlobs)
}

// JSON returns a compact JSON array for the given patterns.
func JSON(patterns []string) string {
	b, err := json.Marshal(patterns)
	if err != nil {
		return "[]"
	}
	return string(b)
}

// JSONPretty returns an indented JSON array for display.
func JSONPretty(patterns []string) string {
	b, err := json.MarshalIndent(patterns, "", "  ")
	if err != nil {
		return "[]"
	}
	return string(b)
}

// InvalidateCache clears the in-memory pattern cache after settings change.
func InvalidateCache() {
	cacheMu.Lock()
	cacheRaw = ""
	cacheParsed = nil
	cacheMu.Unlock()
}

// Match reports whether relPath under projectRoot matches any configured pattern.
func Match(absPath, projectRoot string, patterns []string) bool {
	if len(patterns) == 0 {
		return false
	}
	rel, err := filepath.Rel(projectRoot, absPath)
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

func parseRaw(raw string) []string {
	if raw == "" || raw == "[]" {
		return nil
	}
	var out []string
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil
	}
	for i := range out {
		out[i] = strings.TrimSpace(out[i])
	}
	return out
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
		if strings.HasSuffix(suf, "/**") {
			dir := strings.TrimSuffix(suf, "/**")
			if dir == "" {
				return true
			}
			return rel == dir || strings.HasPrefix(rel, dir+"/") ||
				strings.Contains(rel, "/"+dir+"/") || strings.HasSuffix(rel, "/"+dir)
		}
		if rel == suf {
			return true
		}
		if strings.HasSuffix(rel, "/"+suf) {
			return true
		}
		return strings.Contains(rel, "/"+suf+"/")
	}
	if strings.HasSuffix(pattern, "/**") {
		pre := strings.TrimSuffix(pattern, "/**")
		if pre == "" {
			return true
		}
		return rel == pre || strings.HasPrefix(rel, pre+"/")
	}
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
