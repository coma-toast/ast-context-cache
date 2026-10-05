package watcher

import (
	"database/sql"
	"os"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	deleteFileSymbolsQuery = "DELETE FROM symbols WHERE file = ? AND project_path = ?"
	deleteFileEdgesQuery   = "DELETE FROM edges WHERE source_file = ? AND project_path = ?"
)

// PurgeExcluded removes index rows for files of projectPath that the current
// exclude rules (global globs, skip dirs, .gitignore/.astignore/.stignore, and the
// per-project list) now reject. It only touches the database — no tree walk — so
// it is cheap to run right after an exclude changes. Returns the number purged.
func PurgeExcluded(projectPath string) int {
	projectPath = NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return 0
	}
	indexer.InvalidatePathFilter(projectPath)
	filter := indexer.NewPathFilter(projectPath)
	removed := 0
	for file := range db.GetIndexedFiles(projectPath) {
		if !filter.Excluded(file) {
			continue
		}
		_ = db.IndexWrite(func(tx *sql.Tx) error {
			if _, err := tx.Exec(deleteFileSymbolsQuery, file, projectPath); err != nil {
				return err
			}
			_, err := tx.Exec(deleteFileEdgesQuery, file, projectPath)
			return err
		})
		db.DeleteIndexedFile(file, projectPath)
		removed++
		if PostIndexHook != nil {
			go PostIndexHook(file, projectPath, true)
		}
	}
	if removed > 0 {
		logger.Info("Purged newly excluded files", "files", removed, "project", projectPath)
		realtime.Notify(realtime.IndexCommitted)
	}
	return removed
}

// onIgnoreFileChanged reacts to a .gitignore/.astignore/.stignore edit: the cached
// filter is dropped immediately and newly excluded rows are purged after a short
// debounce. Paths un-ignored by the edit are picked up on the next catch-up
// (watcher restart or index_files), not immediately.
func onIgnoreFileChanged(projectPath string) {
	indexer.InvalidatePathFilter(projectPath)
	// Keyed under projectPath so cancelDebounceTimersForProject also cancels it.
	key := projectPath + string(os.PathSeparator) + "\x00ignore-files"
	debounceMu.Lock()
	defer debounceMu.Unlock()
	if t, ok := debounceTimers[key]; ok {
		stopDebounce(t)
	}
	// Counted in bg like handleFSEvent's timers: stopDebounce (here and in
	// cancelDebounceTimersForProject) calls bg.Done for a timer it cancels, so an
	// uncounted one drove the WaitGroup negative.
	bg.Add(1)
	var t *time.Timer
	t = time.AfterFunc(time.Second, func() {
		defer bg.Done()
		PurgeExcluded(projectPath)
		debounceMu.Lock()
		if debounceTimers[key] == t {
			delete(debounceTimers, key)
		}
		debounceMu.Unlock()
	})
	debounceTimers[key] = t
}
