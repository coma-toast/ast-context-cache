package watcher

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/fsnotify/fsnotify"
)

var (
	mu             sync.Mutex
	activeWatchers = map[string]*fsnotify.Watcher{}
	debounceMu     sync.Mutex
	debounceTimers = map[string]*time.Timer{}
)

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

	log.Printf("File watcher started for %s", projectPath)
}

func handleFSEvent(event fsnotify.Event, projectPath string, w *fsnotify.Watcher) {
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
			log.Printf("Removed symbols for deleted file: %s", path)
			db.LogQuery("file_watcher", map[string]interface{}{"event": "delete", "file": path}, 0, 0, 0, 0, 0, 0,
				float64(time.Since(start).Milliseconds()), projectPath, "")
			if PostIndexHook != nil {
				go PostIndexHook(path, projectPath, true)
			}
		} else {
			n, err := indexer.IndexFile(path, projectPath)
			if err == nil {
				log.Printf("Re-indexed %s: %d symbols", path, n)
				resultJSON, _ := json.Marshal(map[string]interface{}{"file": path, "symbols": n})
				db.LogQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, len(resultJSON), 0, 0, 0, 0, 0,
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
	for project := range activeWatchers {
		watchers = append(watchers, map[string]interface{}{
			"project_path": project,
			"active":       true,
		})
	}
	mu.Unlock()

	return map[string]interface{}{
		"watchers":     watchers,
		"total_active": len(watchers),
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
	log.Printf("Stopped watcher for %s", projectPath)
	return nil
}
