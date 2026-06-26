// Package watcher runs fsnotify-based incremental indexing. Events are filtered with
// indexer.IsCodeFile first (supported code extensions, or .log/.txt when index_log_files is on),
// then MatchWatcherIgnore using watcher_ignore_globs so generated or noisy paths can be skipped
// before debounce/re-index.
package watcher

import (
	"database/sql"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/fsnotify/fsnotify"
)

var (
	mu             sync.Mutex
	activeWatchers = map[string]*fsnotify.Watcher{}
	knownProjects  = map[string]bool{} // true = active, false = stopped
	lastActivity   = map[string]time.Time{}
	debounceMu     sync.Mutex
	debounceTimers = map[string]*time.Timer{}
	catchUpSlots   = make(chan struct{}, 2)
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
	rows, err := db.IndexDB.Query("SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != '' AND project_path != '.'")
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
	mu.Lock()
	if _, exists := activeWatchers[projectPath]; exists {
		mu.Unlock()
		return
	}
	w, err := fsnotify.NewWatcher()
	if err != nil {
		mu.Unlock()
		log.Printf("Watcher error for %s: %v", projectPath, err)
		return
	}
	activeWatchers[projectPath] = w
	knownProjects[projectPath] = true
	lastActivity[projectPath] = time.Now()
	mu.Unlock()

	filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if indexer.ShouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			w.Add(path)
		}
		return nil
	})

	go func() {
		for {
			select {
			case event, ok := <-w.Events:
				if !ok {
					return
				}
				handleFSEvent(event, projectPath, w)
			case err, ok := <-w.Errors:
				if !ok {
					return
				}
				log.Printf("Watcher error: %v", err)
			}
		}
	}()

	go catchUp(projectPath)
	log.Printf("File watcher started for %s", projectPath)
	realtime.Notify(realtime.WatchersChanged)
}

func catchUp(projectPath string) {
	catchUpSlots <- struct{}{}
	defer func() { <-catchUpSlots }()
	indexed := db.GetIndexedFiles(projectPath)
	seen := map[string]bool{}
	stale := 0
	filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if indexer.ShouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if !indexer.IsCodeFile(path) {
			return nil
		}
		seen[path] = true
		if MatchWatcherIgnore(path, projectPath, GetWatcherIgnorePatterns()) {
			return nil
		}
		if idxTime, ok := indexed[path]; ok && !info.ModTime().After(idxTime) {
			return nil
		}
		n, fullT, skelT, err := indexer.IndexFile(path, projectPath)
		if err == nil {
			stale++
			log.Printf("Catch-up re-indexed %s: %d symbols", path, n)
			// Log baseline token counts for analytics; tokens_saved=0 — savings are calculated when querying.
			db.LogQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, db.QueryLogMetrics{TokensUsed: skelT, SymbolBaseline: fullT, FileBaseline: fullT}, projectPath, "")
			if PostIndexHook != nil {
				go PostIndexHook(path, projectPath, false)
			}
		}
		return nil
	})
	removed := 0
		for file := range indexed {
		if !seen[file] {
			_ = db.IndexWrite(func(tx *sql.Tx) error {
				if _, err := tx.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", file, projectPath); err != nil {
					return err
				}
				_, err := tx.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", file, projectPath)
				return err
			})
			db.DeleteIndexedFile(file, projectPath)
			removed++
			if PostIndexHook != nil {
				go PostIndexHook(file, projectPath, true)
			}
		}
	}
	if stale > 0 || removed > 0 {
		log.Printf("Catch-up complete for %s: %d re-indexed, %d removed", projectPath, stale, removed)
	}
	if removed > 0 {
		realtime.Notify(realtime.IndexCommitted)
	}
}

func handleFSEvent(event fsnotify.Event, projectPath string, w *fsnotify.Watcher) {
	mu.Lock()
	lastActivity[projectPath] = time.Now()
	mu.Unlock()
	path := event.Name

	if info, err := os.Stat(path); err == nil && info.IsDir() {
		if event.Has(fsnotify.Create) && !indexer.ShouldSkipDir(info.Name()) {
			w.Add(path)
		}
		return
	}

	if !indexer.IsCodeFile(path) {
		return
	}
	if MatchWatcherIgnore(path, projectPath, GetWatcherIgnorePatterns()) {
		return
	}

	removed := event.Has(fsnotify.Remove) || event.Has(fsnotify.Rename)

	debounceMu.Lock()
	if t, ok := debounceTimers[path]; ok {
		t.Stop()
	}
	debounceTimers[path] = time.AfterFunc(500*time.Millisecond, func() {
		start := time.Now()
		if removed {
			_ = db.IndexWrite(func(tx *sql.Tx) error {
				if _, err := tx.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", path, projectPath); err != nil {
					return err
				}
				_, err := tx.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", path, projectPath)
				return err
			})
			db.DeleteIndexedFile(path, projectPath)
			log.Printf("Removed symbols for deleted file: %s", path)
			db.LogQuery("file_watcher", map[string]interface{}{"event": "delete", "file": path}, db.QueryLogMetrics{}, projectPath, "")
			if PostIndexHook != nil {
				go PostIndexHook(path, projectPath, true)
			}
			realtime.Notify(realtime.IndexCommitted)
		} else {
			n, fullT, skelT, err := indexer.IndexFile(path, projectPath)
			if err == nil {
				log.Printf("Re-indexed %s: %d symbols", path, n)
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
		debounceMu.Lock()
		delete(debounceTimers, path)
		debounceMu.Unlock()
	})
	debounceMu.Unlock()
}

func GetStatus() map[string]interface{} {
	projects := trackedProjectPaths()
	mu.Lock()
	watchers := make([]map[string]interface{}, 0, len(projects))
	active := 0
	for _, project := range projects {
		isActive := knownProjects[project]
		entry := map[string]interface{}{
			"project_path": project,
			"active":       isActive,
		}
		if t, ok := lastActivity[project]; ok {
			entry["last_activity"] = t.Format(time.RFC3339)
		}
		watchers = append(watchers, entry)
		if isActive {
			active++
		}
	}
	mu.Unlock()

	return map[string]interface{}{
		"watchers":     watchers,
		"total_active": active,
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
	log.Printf("Stopped watcher for %s", projectPath)
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
	log.Printf("Deleted watcher for %s", projectPath)
	realtime.Notify(realtime.WatchersChanged)
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
		log.Printf("EnsureWatcher: %s: %v", projectPath, err)
		return
	} else if !info.IsDir() {
		log.Printf("EnsureWatcher: %s is not a directory", projectPath)
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
		timeout := idleTimeout()
		if timeout == 0 {
			continue
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
				log.Printf("Watcher idle timeout for %s", p)
				StopWatcher(p)
			}
		}
	}
}
