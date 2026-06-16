package watcher

import (
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
)

// EnsureDefaultIgnoreGlobs seeds default watcher ignore patterns when unset.
func EnsureDefaultIgnoreGlobs() {
	ignorepatterns.EnsureDefaults()
}

// GetWatcherIgnorePatterns returns glob patterns from settings (with defaults when unset).
func GetWatcherIgnorePatterns() []string {
	return ignorepatterns.List()
}

// MatchWatcherIgnore reports whether absPath matches any ignore pattern relative to projectPath.
func MatchWatcherIgnore(absPath, projectPath string, patterns []string) bool {
	return ignorepatterns.Match(absPath, projectPath, patterns)
}
