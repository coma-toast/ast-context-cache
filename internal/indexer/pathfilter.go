package indexer

import (
	"path/filepath"
	"sync"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/ignorefiles"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
)

// PathFilter decides which paths under a project the indexer and watcher skip,
// on top of ShouldSkipDir: the global watcher ignore globs, the project's
// .gitignore / .astignore / .stignore files, and its per-project exclude list
// (db.ProjectIndexExcludes). Directories it rejects are pruned from walks.
type PathFilter struct {
	root  string
	globs []string
	files *ignorefiles.Matcher
}

// NewPathFilter snapshots the current global globs and per-project excludes and
// returns a filter for one walk (or until invalidated, via CachedPathFilter).
func NewPathFilter(projectPath string) *PathFilter {
	root := filepath.Clean(projectPath)
	return &PathFilter{
		root:  root,
		globs: ignorepatterns.List(),
		files: ignorefiles.New(root, db.ProjectIndexExcludes(root)),
	}
}

// SkipDir reports whether a directory reached during a walk should be pruned
// (filepath.SkipDir). Its ancestors are assumed to have passed already.
func (f *PathFilter) SkipDir(absDir string) bool {
	if filepath.Clean(absDir) == f.root {
		return false
	}
	return ignorepatterns.MatchDir(absDir, f.root, f.globs) || f.files.MatchEntry(absDir, true)
}

// SkipFile reports whether a file reached during a walk should not be indexed.
func (f *PathFilter) SkipFile(absPath string) bool {
	return ignorepatterns.Match(absPath, f.root, f.globs) || f.files.MatchEntry(absPath, false)
}

// IgnoredByFiles reports whether absPath (or an ancestor directory) is excluded
// by the project's ignore files or per-project list. Global globs are not
// consulted, so watcher callers keep their existing MatchWatcherIgnore check.
func (f *PathFilter) IgnoredByFiles(absPath string, isDir bool) bool {
	return f.files.Excluded(absPath, isDir)
}

// Excluded reports whether an already-indexed file should be purged: it matches
// a global glob, sits under a ShouldSkipDir directory, or is ignored (directly or
// via an ancestor) by the project's ignore files or per-project list.
func (f *PathFilter) Excluded(absPath string) bool {
	if ignorepatterns.Match(absPath, f.root, f.globs) || f.files.Excluded(absPath, false) {
		return true
	}
	for dir := filepath.Dir(absPath); dir != f.root && len(dir) > len(f.root); dir = filepath.Dir(dir) {
		if ShouldSkipDir(filepath.Base(dir)) {
			return true
		}
	}
	return false
}

var (
	filterGen   atomic.Int64
	filterMu    sync.Mutex
	filterCache = map[string]cachedFilter{}
)

type cachedFilter struct {
	gen int64
	f   *PathFilter
}

// CachedPathFilter returns a long-lived filter for per-event checks (the file
// watcher). It is rebuilt after InvalidatePathFilter / InvalidateAllPathFilters.
func CachedPathFilter(projectPath string) *PathFilter {
	root := filepath.Clean(projectPath)
	gen := filterGen.Load()
	filterMu.Lock()
	c, ok := filterCache[root]
	filterMu.Unlock()
	if ok && c.gen == gen {
		return c.f
	}
	f := NewPathFilter(root)
	filterMu.Lock()
	filterCache[root] = cachedFilter{gen: gen, f: f}
	filterMu.Unlock()
	return f
}

// InvalidatePathFilter drops the cached filter for one project (an ignore file
// changed, or its per-project exclude list was saved).
func InvalidatePathFilter(projectPath string) {
	filterMu.Lock()
	delete(filterCache, filepath.Clean(projectPath))
	filterMu.Unlock()
}

// InvalidateAllPathFilters drops every cached filter (global globs changed).
func InvalidateAllPathFilters() {
	filterGen.Add(1)
}
