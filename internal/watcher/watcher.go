// Package watcher runs file-watch-based incremental indexing: one FSEvents stream
// per project on macOS, fsnotify (inotify/kqueue) elsewhere (see backend.go).
// Events are filtered with indexer.IsCodeFile first (supported code extensions,
// or .log/.txt when index_log_files is on), then MatchWatcherIgnore using
// watcher_ignore_globs so generated or noisy paths can be skipped before
// debounce/re-index.
package watcher

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/ignorefiles"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/fsnotify/fsnotify"
)

const selectIndexedProjectPathsQuery = "SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != '' AND project_path != '.'"

var (
	mu             sync.Mutex
	activeWatchers = map[string]backend{}
	knownProjects  = map[string]bool{} // true = active, false = stopped
	lastActivity   = map[string]time.Time{}
	debounceMu     sync.Mutex
	debounceTimers = map[string]*time.Timer{}
	catchUpSlots   = make(chan struct{}, 2)

	// bg counts the goroutines watchers start — event loops, catch-ups, and
	// debounce callbacks from when they're scheduled — so tests can wait for
	// all of them before the next test changes what they read.
	bg sync.WaitGroup

	// indexFile is what a debounce callback re-indexes a file with. Tests
	// swap it to hold a callback in flight for as long as they need.
	indexFile = indexer.IndexFile
)

func init() {
	go idleLoop()
}

// PostIndexHook is called after a file is indexed or removed.
// Set this from outside the package to add vector embedding, etc.
var PostIndexHook func(filePath, projectPath string, removed bool)

// NormalizeProjectPath returns a canonical absolute path for watcher map keys.
func NormalizeProjectPath(projectPath string) string {
	projectPath = strings.TrimSpace(projectPath)
	if projectPath == "" {
		return ""
	}
	if abs, err := filepath.Abs(projectPath); err == nil {
		projectPath = abs
	}
	return filepath.Clean(projectPath)
}

// RegisterKnownProject records a project in the dashboard list without starting a watcher.
func RegisterKnownProject(projectPath string) {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	if info, err := os.Stat(projectPath); err != nil || !info.IsDir() {
		return
	}
	mu.Lock()
	if _, ok := knownProjects[projectPath]; !ok {
		knownProjects[projectPath] = false
	}
	mu.Unlock()
}

// RegisterAllKnownProjects registers every indexed repo for the dashboard (inactive).
func RegisterAllKnownProjects() {
	for _, pp := range indexedProjectPaths() {
		RegisterKnownProject(pp)
	}
}

func trackedProjectPaths() []string {
	seen := map[string]bool{}
	add := func(p string) {
		p = NormalizeProjectPath(p)
		if p != "" {
			seen[p] = true
		}
	}
	for _, pp := range indexedProjectPaths() {
		add(pp)
	}
	mu.Lock()
	for pp := range knownProjects {
		add(pp)
	}
	mu.Unlock()
	out := make([]string, 0, len(seen))
	for pp := range seen {
		out = append(out, pp)
	}
	sort.Slice(out, func(i, j int) bool {
		bi := strings.ToLower(filepath.Base(out[i]))
		bj := strings.ToLower(filepath.Base(out[j]))
		if bi != bj {
			return bi < bj
		}
		return out[i] < out[j]
	})
	return out
}

func indexedProjectPaths() []string {
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	rows, err := conn.Query(selectIndexedProjectPathsQuery)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var pp string
		if rows.Scan(&pp) == nil {
			pp = NormalizeProjectPath(pp)
			if pp != "" {
				out = append(out, pp)
			}
		}
	}
	return out
}

func StartWatcher(projectPath string) {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	if reason := WatchRefusal(projectPath); reason != "" {
		logRefusalOnce(projectPath, reason)
		return
	}
	mu.Lock()
	if _, exists := activeWatchers[projectPath]; exists {
		mu.Unlock()
		return
	}
	w, err := newBackend(projectPath)
	if err != nil {
		mu.Unlock()
		logger.Error("Failed to start watcher", "project", projectPath, "error", err)
		return
	}
	activeWatchers[projectPath] = w
	knownProjects[projectPath] = true
	lastActivity[projectPath] = time.Now()
	mu.Unlock()

	indexer.InvalidatePathFilter(projectPath) // ignore files may have changed while unwatched
	filter := indexer.CachedPathFilter(projectPath)
	// A recursive backend already covers every subdirectory; handleFSEvent
	// filters out the ones this walk would have pruned.
	if !w.Recursive() {
		filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() {
				if indexer.ShouldSkipDir(info.Name()) {
					return filepath.SkipDir
				}
				if projectlinks.ShouldSkipDirDuringWalk(path, projectPath) {
					return filepath.SkipDir
				}
				if filter.SkipDir(path) {
					return filepath.SkipDir
				}
				w.Add(path)
			}
			return nil
		})
	}

	bg.Add(2)
	go func() {
		defer bg.Done()
		for {
			select {
			case event, ok := <-w.Events():
				if !ok {
					return
				}
				handleFSEvent(event, projectPath, w)
			case err, ok := <-w.Errors():
				if !ok {
					return
				}
				if errors.Is(err, fsnotify.ErrEventOverflow) {
					scheduleCatchUp(projectPath)
					continue
				}
				logger.Warn("Watcher error", "project", projectPath, "error", err)
			}
		}
	}()

	go func() {
		defer bg.Done()
		catchUp(projectPath)
	}()
	logger.Info("File watcher started", "project", projectPath, "backend", w.Name())
	realtime.Notify(realtime.WatchersChanged)
}

func catchUp(projectPath string) {
	catchUpSlots <- struct{}{}
	defer func() { <-catchUpSlots }()
	// DeleteWatcher doesn't wait for a catch-up already under way, so the
	// catch-up checks instead, and stops before re-indexing a project that was
	// deleted (and purged) while it waited for a slot or walked the tree.
	if projectDeleted(projectPath) {
		return
	}
	indexed := db.GetIndexedFiles(projectPath)
	seen := map[string]bool{}
	stale := 0
	filter := indexer.NewPathFilter(projectPath)
	walkErr := filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if indexer.ShouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			if projectlinks.ShouldSkipDirDuringWalk(path, projectPath) {
				return filepath.SkipDir
			}
			if filter.SkipDir(path) {
				return filepath.SkipDir
			}
			return nil
		}
		if projectlinks.IsUnderLinkedChild(path, projectPath) {
			return nil
		}
		if !indexer.IsCodeFile(path) {
			return nil
		}
		if indexer.SkipSymlinkAlias(path, projectPath, info) {
			return nil
		}
		// Excluded files are left out of seen, so rows indexed before the exclude
		// existed are purged below along with deleted files.
		if filter.SkipFile(path) {
			return nil
		}
		seen[path] = true
		if idxTime, ok := indexed[path]; ok && !info.ModTime().After(idxTime) {
			return nil
		}
		if projectDeleted(projectPath) {
			return filepath.SkipAll
		}
		n, fullT, skelT, err := indexer.IndexFile(path, projectPath)
		if err == nil {
			stale++
			logger.Debug("Catch-up re-indexed file", "file", path, "symbols", n)
			// Log baseline token counts for analytics; tokens_saved=0 — savings are calculated when querying.
			db.LogQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, db.QueryLogMetrics{TokensUsed: skelT, SymbolBaseline: fullT, FileBaseline: fullT}, projectPath, "")
			if PostIndexHook != nil {
				go PostIndexHook(path, projectPath, false)
			}
		}
		return nil
	})
	if projectDeleted(projectPath) {
		return
	}
	removed := 0
	for _, file := range catchUpRemovals(projectPath, indexed, seen, walkErr == nil) {
		if err := removeFileFromIndex(file, projectPath); err != nil {
			logger.Warn("Failed to purge file during catch-up", "file", file, "error", err)
			continue
		}
		removed++
		if PostIndexHook != nil {
			go PostIndexHook(file, projectPath, true)
		}
	}
	if stale > 0 || removed > 0 {
		logger.Info("Catch-up complete", "project", projectPath, "reindexed", stale, "removed", removed)
	}
	if removed > 0 {
		realtime.Notify(realtime.IndexCommitted)
	}
}

// removeFileFromIndex purges a removed file's rows (symbols, edges, code vectors,
// summaries, pending embeds, indexed_files) in one index write; see
// indexer.PurgeFile. If the write fails (a WAL quiesce gates index writes),
// nothing is deleted and the file stays recorded as indexed, so the next
// catch-up scan retries it.
func removeFileFromIndex(file, projectPath string) error {
	return indexer.PurgeFile(file, projectPath)
}

// catchUpRemovals lists index entries catch-up must purge. Candidates are every
// file with rows for the project (not only indexed_files, so rows that lost or
// never had an indexed_files entry are reconciled too). A tracked file under the
// project that a complete walk did not visit is removed, as before; anything else
// (untracked rows, paths outside the project, or any file after an aborted walk)
// is removed only when it is provably gone from disk.
func catchUpRemovals(projectPath string, indexed map[string]time.Time, seen map[string]bool, walkComplete bool) []string {
	candidates := map[string]bool{}
	for f := range indexed {
		candidates[f] = true
	}
	for _, f := range indexer.ProjectFilesInIndex(projectPath) {
		candidates[f] = true
	}
	prefix := projectPath + string(filepath.Separator)
	var out []string
	for f := range candidates {
		if seen[f] || !filepath.IsAbs(f) {
			continue
		}
		_, tracked := indexed[f]
		if tracked && walkComplete && strings.HasPrefix(f, prefix) {
			out = append(out, f)
			continue
		}
		if indexer.FileGone(f) {
			out = append(out, f)
		}
	}
	sort.Strings(out)
	return out
}

func handleFSEvent(event fsnotify.Event, projectPath string, w backend) {
	path := event.Name
	// Both checked before counting as activity. A recursive backend reports
	// the whole tree: .git/ and node_modules/ churn constantly (which would
	// keep the watcher from ever going idle), and linked child projects have
	// watchers of their own.
	if underSkippedDir(path, projectPath) || projectlinks.IsUnderLinkedChild(path, projectPath) {
		return
	}
	mu.Lock()
	lastActivity[projectPath] = time.Now()
	mu.Unlock()

	if info, err := os.Stat(path); err == nil && info.IsDir() {
		if event.Has(fsnotify.Create) && !indexer.ShouldSkipDir(info.Name()) && !projectlinks.ShouldSkipDirDuringWalk(path, projectPath) &&
			!indexer.CachedPathFilter(projectPath).IgnoredByFiles(path, true) {
			w.Add(path)
		}
		return
	}
	if ignorefiles.IsIgnoreFileName(filepath.Base(path)) {
		onIgnoreFileChanged(projectPath)
		return
	}

	// A repo's own scripts/code-mode/ manifest or script files were cached
	// on first use with no invalidation wiring at all — editing them needed a
	// full ast-mcp restart to take effect. manifest.json isn't necessarily a
	// "code file" IsCodeFile would recognize, so this check runs before that
	// filter, not after it.
	if codescripts.IsRepoScriptPath(path, projectPath) {
		codescripts.InvalidateRepoCache(projectPath)
	}

	if !indexer.IsCodeFile(path) {
		return
	}
	if MatchWatcherIgnore(path, projectPath, GetWatcherIgnorePatterns()) {
		return
	}
	if indexer.CachedPathFilter(projectPath).IgnoredByFiles(path, false) {
		return
	}

	removed := event.Has(fsnotify.Remove) || event.Has(fsnotify.Rename)

	key := debounceKey(projectPath, path)
	debounceMu.Lock()
	if t, ok := debounceTimers[key]; ok {
		stopDebounce(t)
	}
	bg.Add(1)
	var t *time.Timer
	t = time.AfterFunc(500*time.Millisecond, func() {
		defer bg.Done()
		start := time.Now()
		if removed {
			if err := removeFileFromIndex(path, projectPath); err != nil {
				// The file is still recorded as indexed, so the next catch-up scan
				// retries the removal.
				logger.Warn("Failed to remove deleted file from the index", "file", path, "error", err)
			} else {
				logger.Debug("Removed symbols for deleted file", "file", path)
				db.LogQuery("file_watcher", map[string]interface{}{"event": "delete", "file": path}, db.QueryLogMetrics{}, projectPath, "")
				if PostIndexHook != nil {
					go PostIndexHook(path, projectPath, true)
				}
				realtime.Notify(realtime.IndexCommitted)
			}
		} else {
			n, fullT, skelT, err := indexFile(path, projectPath)
			if err == nil {
				logger.Debug("Re-indexed file", "file", path, "symbols", n)
				resultJSON, _ := json.Marshal(map[string]interface{}{"file": path, "symbols": n})
				// Log baseline token counts for analytics; tokens_saved=0 — savings are calculated when querying.
				db.LogQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, db.QueryLogMetrics{
					ResultChars: len(resultJSON), TokensUsed: skelT, SymbolBaseline: fullT, FileBaseline: fullT,
					DurationMs: float64(time.Since(start).Milliseconds()),
				}, projectPath, "")
				if PostIndexHook != nil {
					go PostIndexHook(path, projectPath, false)
				}
			}
		}
		// Stop is a no-op once a timer has fired, so an event that arrived
		// while this ran may have queued a newer timer under key. Leave that
		// one for cancelDebounceTimersForProject to find.
		debounceMu.Lock()
		if debounceTimers[key] == t {
			delete(debounceTimers, key)
		}
		debounceMu.Unlock()
	})
	debounceTimers[key] = t
	debounceMu.Unlock()
}

func GetStatus() map[string]interface{} {
	projects := trackedProjectPaths()
	roots := containerRoots()
	mu.Lock()
	watchers := make([]map[string]interface{}, 0, len(projects))
	active, osWatches := 0, 0
	for _, project := range projects {
		isActive := knownProjects[project]
		entry := map[string]interface{}{
			"project_path": project,
			"active":       isActive,
		}
		if t, ok := lastActivity[project]; ok {
			entry["last_activity"] = t.Format(time.RFC3339)
		}
		if w, ok := activeWatchers[project]; ok {
			n := w.OSWatches()
			entry["backend"] = w.Name()
			entry["os_watches"] = n
			osWatches += n
		} else if reason := watchRefusalFor(project, roots); reason != "" {
			entry["blocked_reason"] = reason
		}
		watchers = append(watchers, entry)
		if isActive {
			active++
		}
	}
	mu.Unlock()

	return map[string]interface{}{
		"watchers":         watchers,
		"total_active":     active,
		"backend":          DefaultBackendName(),
		"os_watches":       osWatches,
		"file_descriptors": FDStatus(),
	}
}

func StopWatcher(projectPath string) error {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return nil
	}
	mu.Lock()
	defer mu.Unlock()

	w, exists := activeWatchers[projectPath]
	if !exists {
		if knownProjects[projectPath] {
			knownProjects[projectPath] = false
			realtime.Notify(realtime.WatchersChanged)
		}
		return nil
	}

	w.Close()
	delete(activeWatchers, projectPath)
	knownProjects[projectPath] = false
	logger.Info("Stopped watcher", "project", projectPath)
	realtime.Notify(realtime.WatchersChanged)
	return nil
}

func DeleteWatcher(projectPath string) {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	mu.Lock()
	if w, exists := activeWatchers[projectPath]; exists {
		w.Close()
		delete(activeWatchers, projectPath)
	}
	delete(knownProjects, projectPath)
	delete(lastActivity, projectPath)
	mu.Unlock()
	cancelDebounceTimersForProject(projectPath)
	logger.Info("Deleted watcher", "project", projectPath)
	realtime.Notify(realtime.WatchersChanged)
}

// StopAll deletes every watcher, cancels every pending debounce, and waits for
// the goroutines they started (event loops, catch-ups, debounced re-indexes) to
// return, so nothing of ours is still using the db pools when they are closed.
func StopAll() {
	mu.Lock()
	projects := make([]string, 0, len(activeWatchers))
	for p := range activeWatchers {
		projects = append(projects, p)
	}
	mu.Unlock()
	for _, p := range projects {
		DeleteWatcher(p)
	}
	debounceMu.Lock()
	for key, t := range debounceTimers {
		stopDebounce(t)
		delete(debounceTimers, key)
	}
	debounceMu.Unlock()
	bg.Wait()
}

// cancelDebounceTimersForProject stops and forgets any pending debounce timer
// for a file under projectPath. Without this, a timer queued by handleFSEvent
// just before a project is deleted can still fire ~500ms later and re-index
// (or delete symbols for) a file the delete just purged — with no watcher left
// to have caused it, since the timer already captured path/projectPath in its
// closure before DeleteWatcher ran.
func cancelDebounceTimersForProject(projectPath string) {
	if projectPath == "" {
		return
	}
	debounceMu.Lock()
	for key, t := range debounceTimers {
		if debounceKeyOwnedBy(key, projectPath) {
			stopDebounce(t)
			delete(debounceTimers, key)
		}
	}
	debounceMu.Unlock()
}

// stopDebounce cancels a debounce timer. If that keeps its callback from ever
// running, it also drops the callback's count in bg.
func stopDebounce(t *time.Timer) {
	if t.Stop() {
		bg.Done()
	}
}

// projectDeleted reports whether DeleteWatcher has forgotten the project.
// StopWatcher (idle unload) keeps it known, and leaves a catch-up running.
func projectDeleted(projectPath string) bool {
	mu.Lock()
	defer mu.Unlock()
	_, known := knownProjects[projectPath]
	return !known
}

// debounceKey keys a file's pending re-index by project as well as path.
// Nested projects (a space root and a repo in it) both watch the file, and a
// path-only key let one project's event replace the other's timer, so only
// one of them ever saw the change.
func debounceKey(projectPath, path string) string {
	return projectPath + "\x00" + path
}

// debounceKeyOwnedBy matches debounceKey keys and per-project keys of the
// form "<project>/\x00<name>" by their owning project exactly, so deleting a
// space root leaves the timers of a repo inside it alone. Keys without a NUL
// are bare file paths and match by path prefix.
func debounceKeyOwnedBy(key, projectPath string) bool {
	if i := strings.IndexByte(key, 0); i >= 0 {
		return strings.TrimSuffix(key[:i], string(os.PathSeparator)) == projectPath
	}
	return key == projectPath || strings.HasPrefix(key, projectPath+string(os.PathSeparator))
}

// IsActive reports whether a watcher is currently running for the project.
func IsActive(projectPath string) bool {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return false
	}
	mu.Lock()
	defer mu.Unlock()
	_, running := activeWatchers[projectPath]
	return running
}

// EnsureWatcher starts a watcher for the project if one isn't already running.
// Bumps lastActivity when already running so MCP/dashboard use resets idle timeout.
func EnsureWatcher(projectPath string) {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	mu.Lock()
	_, running := activeWatchers[projectPath]
	if running {
		lastActivity[projectPath] = time.Now()
		mu.Unlock()
		return
	}
	mu.Unlock()
	if info, err := os.Stat(projectPath); err != nil {
		logger.Warn("Failed to stat project for watcher", "project", projectPath, "error", err)
		return
	} else if !info.IsDir() {
		logger.Warn("Project for watcher is not a directory", "project", projectPath)
		return
	}
	StartWatcher(projectPath)
}

func shouldStopForIdle(project string, now time.Time, timeout time.Duration) bool {
	if timeout == 0 {
		return false
	}
	mu.Lock()
	defer mu.Unlock()
	if !knownProjects[project] {
		return false
	}
	if db.IsPinnedProject(project) {
		return false
	}
	t, ok := lastActivity[project]
	return ok && now.Sub(t) > timeout
}

// anyWatcherActive reports whether any project has a running watcher.
func anyWatcherActive() bool {
	mu.Lock()
	defer mu.Unlock()
	for _, isActive := range knownProjects {
		if isActive {
			return true
		}
	}
	return false
}

func idleTimeout() time.Duration {
	val := db.GetSetting("idle_unload_minutes", "1")
	mins, err := strconv.Atoi(val)
	if err != nil || mins < 0 {
		mins = 1
	}
	if mins == 0 {
		return 0
	}
	return time.Duration(mins) * time.Minute
}

func idleLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		idleTick()
	}
}

// idleTick stops watchers that have seen no activity for idleTimeout.
func idleTick() {
	checkFDPressure()
	// With no watcher running there's nothing to stop, so don't touch the
	// db. This loop starts in init() and runs in every binary that imports
	// watcher, while tests open and close db's package-global pools with no
	// lock this loop could share.
	if !anyWatcherActive() {
		return
	}
	timeout := idleTimeout()
	if timeout == 0 {
		return
	}
	mu.Lock()
	now := time.Now()
	var toStop []string
	for project, isActive := range knownProjects {
		if !isActive {
			continue
		}
		if db.IsPinnedProject(project) {
			continue
		}
		if t, ok := lastActivity[project]; ok && now.Sub(t) > timeout {
			toStop = append(toStop, project)
		}
	}
	mu.Unlock()
	for _, p := range toStop {
		if shouldStopForIdle(p, now, timeout) {
			logger.Info("Watcher idle timeout", "project", p)
			StopWatcher(p)
		}
	}
}
