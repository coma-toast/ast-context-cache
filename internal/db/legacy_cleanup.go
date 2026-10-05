package db

import (
	"os"
	"path/filepath"
	"time"
)

// legacyDBNames are database filenames older releases (or a stray `sqlite3 <name>` that
// opened a path and never wrote) left in the data directory. No current code opens them.
var legacyDBNames = []string{"ast-cache.db", "ast.db", "astcache.db"}

// sqliteSidecars are the files SQLite keeps next to a database; a legacy file with any of
// them present may still hold data (a zero-byte main file plus a WAL is not empty).
var sqliteSidecars = []string{"-wal", "-shm", "-journal"}

// legacyMinIdle is how long a legacy file must have gone unmodified before it is swept.
// SQLite creates a zero-byte file on open and writes the header only on the first write,
// so a just-created empty file may belong to another process (e.g. a second server with
// DB_PATH pointing at a legacy name) that is about to write to it. The strays from the
// field report were months old; an hour is plenty to rule out a racing opener.
var legacyMinIdle = time.Hour

// removeEmptyLegacyDBs deletes zero-byte legacy database files from dir and returns the
// names removed. It never touches the active index/context/usage databases (including a
// DB_PATH that happens to use a legacy name), non-regular files (symlinks included),
// non-empty files, a file modified within legacyMinIdle, or a file that still has a
// SQLite sidecar.
func removeEmptyLegacyDBs(dir string) []string {
	active := map[string]bool{}
	for _, p := range []string{indexDBPath(), contextDBPath(), usageDBPath()} {
		if abs, err := filepath.Abs(p); err == nil {
			active[abs] = true
		}
	}
	var removed []string
	for _, name := range legacyDBNames {
		p := filepath.Join(dir, name)
		abs, err := filepath.Abs(p)
		if err != nil || active[abs] {
			continue
		}
		fi, err := os.Lstat(p)
		if err != nil || !fi.Mode().IsRegular() || fi.Size() != 0 || time.Since(fi.ModTime()) < legacyMinIdle || hasSQLiteSidecar(p) {
			continue
		}
		if err := os.Remove(p); err != nil {
			logger.Warn("Failed to remove empty legacy database", "path", p, "error", err)
			continue
		}
		removed = append(removed, name)
	}
	if len(removed) > 0 {
		logger.Info("Removed empty legacy database files", "count", len(removed), "dir", dir, "files", removed)
	}
	return removed
}

func hasSQLiteSidecar(p string) bool {
	for _, suffix := range sqliteSidecars {
		if _, err := os.Lstat(p + suffix); err == nil {
			return true
		}
	}
	return false
}
