package db

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// DataDirMoveSnapshot is a point-in-time view of a data-directory move for the dashboard.
type DataDirMoveSnapshot struct {
	Active     bool
	Done       bool
	Phase      string
	TargetDir  string
	StartedAt  time.Time
	FinishedAt time.Time
	Error      string
	// Recreated lists db filenames (index.db, context.db, usage.db) whose source file
	// was missing at move time — e.g. a USB drive that disconnected and came back with
	// the file gone. Each is started fresh and empty at the target instead of copying
	// whatever SQLite would otherwise auto-create at the dead source path.
	Recreated []string
	// Kept lists db filenames that already existed (non-empty) at the target and were
	// left as-is instead of being overwritten by a copy from source — e.g. moving back
	// to a USB drive that already holds a database from an earlier move.
	Kept []string
}

var (
	dataDirMoveMu sync.RWMutex
	dataDirMove   DataDirMoveSnapshot
)

// GetDataDirMoveSnapshot returns the current data-directory move state for the dashboard.
func GetDataDirMoveSnapshot() DataDirMoveSnapshot {
	dataDirMoveMu.RLock()
	defer dataDirMoveMu.RUnlock()
	return dataDirMove
}

func setDataDirMove(s DataDirMoveSnapshot) {
	dataDirMoveMu.Lock()
	dataDirMove = s
	dataDirMoveMu.Unlock()
	realtime.Notify(realtime.Settings)
}

// StartDataDirMove validates target and, if valid, copies all three databases into it in
// the background via SQLite's VACUUM INTO (a consistent point-in-time copy that runs
// safely while the source databases stay open and serving normal traffic). On success it
// records target in the location override file; the new location takes effect on the next
// restart. Existing files at the current location are never modified or removed.
func StartDataDirMove(target string) (started bool, errMsg string) {
	if GetDataDirMoveSnapshot().Active {
		return false, "a data directory move is already in progress"
	}
	target = strings.TrimSpace(target)
	if target == "" {
		return false, "target directory is required"
	}
	if !filepath.IsAbs(target) {
		return false, "target directory must be an absolute path"
	}
	target = filepath.Clean(target)
	if target == cacheDir() {
		return false, "target directory is already the current data directory"
	}
	if err := os.MkdirAll(target, 0o755); err != nil {
		return false, fmt.Sprintf("cannot create target directory: %v", err)
	}
	probe := filepath.Join(target, ".astcache-write-test")
	if err := os.WriteFile(probe, []byte("ok"), 0o644); err != nil {
		return false, fmt.Sprintf("target directory is not writable: %v", err)
	}
	os.Remove(probe)

	setDataDirMove(DataDirMoveSnapshot{Active: true, TargetDir: target, StartedAt: time.Now(), Phase: "starting"})
	go runDataDirMove(target)
	return true, ""
}

func runDataDirMove(target string) {
	type step struct {
		pool       *sql.DB
		filename   string
		label      string
		sourcePath string
	}
	steps := []step{
		{IndexDB, indexFile, "copying index.db", indexDBPath()},
		{ContextDB, contextFile, "copying context.db", contextDBPath()},
		{DB, usageFile, "copying usage.db", usageDBPath()},
	}
	var recreated, kept []string
	for _, s := range steps {
		snap := GetDataDirMoveSnapshot()
		snap.Phase = s.label
		setDataDirMove(snap)

		destPath := filepath.Join(target, s.filename)

		if fi, err := os.Stat(destPath); err == nil && fi.Size() > 0 {
			// A database already sits at the target — most likely this is a drive
			// that was moved to before (e.g. reconnecting a USB drive that already
			// holds a full copy from an earlier move). Switch to using it rather than
			// clobbering it with a fresh copy from source.
			log.Printf("data dir move: %s already exists at %s (%d bytes) — keeping it instead of overwriting", s.filename, destPath, fi.Size())
			kept = append(kept, s.filename)
			continue
		}
		// A zero-byte leftover (e.g. from a previously interrupted move) isn't worth
		// keeping — clear it so the copy/create-fresh path below starts clean.
		os.Remove(destPath)

		_, statErr := os.Stat(s.sourcePath)
		// Nothing usable to copy from — either the pool itself isn't open (e.g. the
		// WAL-maintenance quiesce nil'd IndexDB and couldn't reopen it because the
		// drive holding it went missing), or the source file is gone even though the
		// pool handle is still open (the drive disconnected and came back empty, or
		// was swapped for a different one at the same path). In the latter case a
		// VACUUM INTO would have SQLite silently (re)create an empty database at
		// sourcePath first and copy that, losing whatever was there without saying
		// so. Either way, start fresh at the target instead and say so plainly.
		if s.pool == nil || statErr != nil {
			if createErr := createEmptyDB(destPath); createErr != nil {
				finishDataDirMove(fmt.Errorf("%s: source unavailable and could not create a fresh database: %w", s.label, createErr))
				return
			}
			log.Printf("data dir move: %s unavailable (pool open=%v, source stat err=%v) — created a fresh, empty database at %s instead of copying", s.filename, s.pool != nil, statErr, destPath)
			recreated = append(recreated, s.filename)
			continue
		}

		if _, err := s.pool.Exec(`VACUUM INTO ?`, destPath); err != nil {
			os.Remove(destPath)
			finishDataDirMove(fmt.Errorf("%s: %w", s.label, err))
			return
		}
	}

	snap := GetDataDirMoveSnapshot()
	snap.Phase = "finalizing"
	setDataDirMove(snap)

	if err := os.WriteFile(locationOverridePath(), []byte(target+"\n"), 0o644); err != nil {
		finishDataDirMove(fmt.Errorf("finalizing: writing %s: %w", locationOverridePath(), err))
		return
	}

	switch {
	case len(recreated) > 0 || len(kept) > 0:
		log.Printf("data dir move: finished at %s (recreated: %s; kept existing: %s) — restart ast-mcp to use the new location", target, joinOrNone(recreated), joinOrNone(kept))
	default:
		log.Printf("data dir move: copied index.db, context.db, and usage.db to %s — restart ast-mcp to use the new location", target)
	}
	snap = GetDataDirMoveSnapshot()
	snap.Active = false
	snap.Done = true
	snap.Phase = "done"
	snap.FinishedAt = time.Now()
	snap.Recreated = recreated
	snap.Kept = kept
	setDataDirMove(snap)
}

func joinOrNone(items []string) string {
	if len(items) == 0 {
		return "none"
	}
	return strings.Join(items, ", ")
}

// createEmptyDB opens (creating if needed) a fresh SQLite database file at path and
// closes it immediately — used when a move's source file is missing and there is
// nothing to copy.
func createEmptyDB(path string) error {
	conn, err := sql.Open("sqlite3", path)
	if err != nil {
		return err
	}
	defer conn.Close()
	return conn.Ping()
}

func finishDataDirMove(err error) {
	log.Printf("data dir move: failed: %v", err)
	snap := GetDataDirMoveSnapshot()
	snap.Active = false
	snap.Done = false
	snap.Phase = "error"
	snap.Error = err.Error()
	snap.FinishedAt = time.Now()
	setDataDirMove(snap)
}
