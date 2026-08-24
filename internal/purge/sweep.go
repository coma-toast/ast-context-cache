package purge

import (
	"log"
	"os"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const (
	// sweepInterval is deliberately slower than the watcher's 30s idle tick: this
	// sweep destroys data, so it trades reaction time for caution.
	sweepInterval = 5 * time.Minute
	// missesBeforePurge is how many consecutive sweeps a directory must be absent
	// before its data is dropped (~15 minutes), so an unmounted volume or a wtg
	// operation mid-flight does not trigger a purge.
	missesBeforePurge = 3
)

var (
	missMu     sync.Mutex
	missCounts = map[string]int{}
)

// StartDeletedProjectSweep runs the deleted-project sweep on a ticker.
func StartDeletedProjectSweep() {
	go func() {
		ticker := time.NewTicker(sweepInterval)
		defer ticker.Stop()
		for range ticker.C {
			SweepDeletedProjects()
		}
	}()
}

// SweepDeletedProjects purges every known project whose directory has been
// missing for missesBeforePurge consecutive sweeps. Returns the paths purged.
//
// Pinned projects are not exempt: pinning protects a project from being evicted
// while it is in use, and there is nothing left to protect once the directory is
// gone.
func SweepDeletedProjects() []string {
	due := recordScan(KnownProjectPaths(), dirExists)
	var purged []string
	for _, p := range due {
		watcher.DeleteWatcher(p)
		if err := ProjectData(p); err != nil {
			log.Printf("purge: %s: %v", p, err)
			continue
		}
		log.Printf("purged data for deleted project: %s", p)
		purged = append(purged, p)
	}
	if len(purged) > 0 {
		realtime.Notify(realtime.IndexHealth)
	}
	return purged
}

// recordScan updates the consecutive-miss counters for one sweep and returns the
// paths that have now been missing long enough to purge. It is pure bookkeeping —
// no database or filesystem work — so the debounce is testable without a ticker.
func recordScan(paths []string, exists func(string) bool) []string {
	missMu.Lock()
	defer missMu.Unlock()

	seen := make(map[string]bool, len(paths))
	var due []string
	for _, p := range paths {
		if p == "" {
			continue
		}
		seen[p] = true
		if exists(p) {
			// Reappeared (or never gone): start over.
			delete(missCounts, p)
			continue
		}
		missCounts[p]++
		if missCounts[p] >= missesBeforePurge {
			delete(missCounts, p)
			due = append(due, p)
		}
	}
	// Forget counters for paths that are no longer tracked at all.
	for p := range missCounts {
		if !seen[p] {
			delete(missCounts, p)
		}
	}
	return due
}

// ResetMissCounts clears debounce state (tests, and after a manual purge).
func ResetMissCounts() {
	missMu.Lock()
	defer missMu.Unlock()
	missCounts = map[string]int{}
}

// MissCount reports consecutive misses recorded for a path.
func MissCount(projectPath string) int {
	missMu.Lock()
	defer missMu.Unlock()
	return missCounts[watcher.NormalizeProjectPath(projectPath)]
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// KnownProjectPaths returns every project path the server tracks: watched
// projects plus projects that are indexed but not currently watched.
func KnownProjectPaths() []string {
	seen := map[string]bool{}
	var out []string
	add := func(p string) {
		p = watcher.NormalizeProjectPath(p)
		if p == "" || p == "." || seen[p] {
			return
		}
		seen[p] = true
		out = append(out, p)
	}

	status := watcher.GetStatus()
	if watchers, ok := status["watchers"].([]map[string]interface{}); ok {
		for _, w := range watchers {
			pp, _ := w["project_path"].(string)
			add(pp)
		}
	}
	for _, p := range projectlinks.IndexedProjectPaths() {
		add(p)
	}
	for _, p := range indexedFileProjectPaths() {
		add(p)
	}
	return out
}

// indexedFileProjectPaths covers projects whose files are recorded but whose
// symbols were all removed, which projectlinks.IndexedProjectPaths would miss.
func indexedFileProjectPaths() []string {
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	rows, err := conn.Query(`SELECT DISTINCT project_path FROM indexed_files WHERE project_path IS NOT NULL AND project_path != ''`)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var p string
		if rows.Scan(&p) == nil {
			out = append(out, p)
		}
	}
	return out
}
