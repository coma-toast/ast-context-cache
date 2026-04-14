package watcher

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/fsnotify/fsnotify"
)

var (
	mu             sync.Mutex
	activeWatchers = map[string]*fsnotify.Watcher{}
	knownProjects  = map[string]bool{} // true = active, false = stopped
	lastActivity   = map[string]time.Time{}
	debounceMu     sync.Mutex
	debounceTimers = map[string]*time.Timer{}
)

func init() {
	go idleLoop()
}

// PostIndexHook is called after a file is indexed or removed.
// Set this from outside the package to add vector embedding, etc.
var PostIndexHook func(filePath, projectPath string, removed bool)

func StartWatcher(projectPath string) {
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
}

func catchUp(projectPath string) {
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
		if idxTime, ok := indexed[path]; ok && !info.ModTime().After(idxTime) {
			return nil
		}
		n, fullT, skelT, err := indexer.IndexFile(path, projectPath)
		if err == nil {
			stale++
			log.Printf("Catch-up re-indexed %s: %d symbols", path, n)
			// Log baseline token counts for analytics; tokens_saved=0 — savings are calculated when querying.
			db.LogQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, 0, 0, skelT, 0, fullT, fullT, 0, projectPath, "")
			if PostIndexHook != nil {
				go PostIndexHook(path, projectPath, false)
			}
		}
		return nil
	})
	removed := 0
	for file := range indexed {
		if !seen[file] {
			db.DB.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", file, projectPath)
			db.DB.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", file, projectPath)
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

	removed := event.Has(fsnotify.Remove) || event.Has(fsnotify.Rename)

	debounceMu.Lock()
	if t, ok := debounceTimers[path]; ok {
		t.Stop()
	}
	debounceTimers[path] = time.AfterFunc(500*time.Millisecond, func() {
		start := time.Now()
		if removed {
			db.DB.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", path, projectPath)
			db.DB.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", path, projectPath)
			db.DeleteIndexedFile(path, projectPath)
			log.Printf("Removed symbols for deleted file: %s", path)
			db.LogQuery("file_watcher", map[string]interface{}{"event": "delete", "file": path}, 0, 0, 0, 0, 0, 0,
				float64(time.Since(start).Milliseconds()), projectPath, "")
			if PostIndexHook != nil {
				go PostIndexHook(path, projectPath, true)
			}
		} else {
			n, fullT, skelT, err := indexer.IndexFile(path, projectPath)
			if err == nil {
				log.Printf("Re-indexed %s: %d symbols", path, n)
				resultJSON, _ := json.Marshal(map[string]interface{}{"file": path, "symbols": n})
				// Log baseline token counts for analytics; tokens_saved=0 — savings are calculated when querying.
				db.LogQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, len(resultJSON), 0, skelT, 0, fullT, fullT,
					float64(time.Since(start).Milliseconds()), projectPath, "")
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
	mu.Lock()
	watchers := []map[string]interface{}{}
	active := 0
	for project, isActive := range knownProjects {
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
	mu.Lock()
	defer mu.Unlock()

	w, exists := activeWatchers[projectPath]
	if !exists {
		return nil
	}

	w.Close()
	delete(activeWatchers, projectPath)
	knownProjects[projectPath] = false
	log.Printf("Stopped watcher for %s", projectPath)
	return nil
}

func DeleteWatcher(projectPath string) {
	mu.Lock()
	if w, exists := activeWatchers[projectPath]; exists {
		w.Close()
		delete(activeWatchers, projectPath)
	}
	delete(knownProjects, projectPath)
	delete(lastActivity, projectPath)
	mu.Unlock()
	log.Printf("Deleted watcher for %s", projectPath)
}

// EnsureWatcher starts a watcher for the project if one isn't already running.
// Works for both previously-known stopped watchers and projects that have
// indexed data but no watcher yet.
func EnsureWatcher(projectPath string) {
	if projectPath == "" {
		return
	}
	mu.Lock()
	_, active := activeWatchers[projectPath]
	mu.Unlock()
	if active {
		return
	}
	if info, err := os.Stat(projectPath); err == nil && info.IsDir() {
		go StartWatcher(projectPath)
	}
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
			log.Printf("Watcher idle timeout for %s", p)
			StopWatcher(p)
		}
	}
}
