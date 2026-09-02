package db

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	indexFile   = "index.db"
	contextFile = "context.db"
	usageFile   = "usage.db"
)

// locationOverridePath is a fixed sibling of the default data directory (never itself
// relocated) that stores a user-chosen data directory set via the Settings UI "Move data
// directory" action. The DB_PATH env var still takes precedence when set.
func locationOverridePath() string {
	home := os.Getenv("HOME")
	if home == "" {
		return ".astcache.location"
	}
	return filepath.Join(home, ".astcache.location")
}

func dataDirOverride() string {
	data, err := os.ReadFile(locationOverridePath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func defaultCacheDir() string {
	home := os.Getenv("HOME")
	if home == "" {
		return ".astcache"
	}
	return filepath.Join(home, ".astcache")
}

// cacheDir resolves the data directory for path-building purposes: DB_PATH env var >
// the Settings-configured override file > the default ~/.astcache. It does not validate
// the result — see ResolveDataDir for the startup fail-fast check.
func cacheDir() string {
	if p := os.Getenv("DB_PATH"); p != "" {
		return filepath.Dir(p)
	}
	if d := dataDirOverride(); d != "" {
		return d
	}
	return defaultCacheDir()
}

// ResolveDataDir resolves the data directory the same way cacheDir does, but validates a
// Settings-configured override actually exists and is writable, so Init can fail fast
// with a clear error (e.g. an unplugged USB drive) instead of silently falling back or
// creating a fresh database at the default location.
func ResolveDataDir() (string, error) {
	if p := os.Getenv("DB_PATH"); p != "" {
		return filepath.Dir(p), nil
	}
	d := dataDirOverride()
	if d == "" {
		return defaultCacheDir(), nil
	}
	info, err := os.Stat(d)
	if err != nil {
		return "", fmt.Errorf("data directory %s: %w", d, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("data directory %s is not a directory", d)
	}
	probe := filepath.Join(d, ".astcache-write-test")
	if err := os.WriteFile(probe, []byte("ok"), 0o644); err != nil {
		return "", fmt.Errorf("data directory %s is not writable: %w", d, err)
	}
	os.Remove(probe)
	return d, nil
}

func indexDBPath() string {
	if p := os.Getenv("DB_PATH"); p != "" {
		dir := filepath.Dir(p)
		return filepath.Join(dir, indexFile)
	}
	return filepath.Join(cacheDir(), indexFile)
}

func contextDBPath() string {
	if p := os.Getenv("DB_PATH"); p != "" {
		dir := filepath.Dir(p)
		return filepath.Join(dir, contextFile)
	}
	return filepath.Join(cacheDir(), contextFile)
}

func usageDBPath() string {
	if p := os.Getenv("DB_PATH"); p != "" {
		return p
	}
	return filepath.Join(cacheDir(), usageFile)
}

func dbPath() string {
	return usageDBPath()
}

// GetDBPath returns the usage database path (legacy name; settings and query log).
func GetDBPath() string {
	return usageDBPath()
}

// GetIndexDBPath returns the code index database path.
func GetIndexDBPath() string {
	return indexDBPath()
}

// GetDataDir returns the directory currently holding the three databases.
func GetDataDir() string {
	return cacheDir()
}

// DataDirSizeBytes sums the on-disk size of the three databases (plus WAL/SHM sidecars)
// under the current data directory.
func DataDirSizeBytes() int64 {
	var total int64
	for _, p := range []string{indexDBPath(), contextDBPath(), usageDBPath()} {
		for _, suffix := range []string{"", "-wal", "-shm"} {
			if fi, err := os.Stat(p + suffix); err == nil {
				total += fi.Size()
			}
		}
	}
	return total
}

// MainDBFilesSizeBytes sums the size of the three main database files only, excluding
// WAL/SHM sidecars. WAL size fluctuates independently of page reclamation (checkpoints,
// in-flight writes) and would make a before/after VACUUM comparison noisy; the main file
// is what VACUUM actually shrinks.
func MainDBFilesSizeBytes() int64 {
	var total int64
	for _, p := range []string{indexDBPath(), contextDBPath(), usageDBPath()} {
		if fi, err := os.Stat(p); err == nil {
			total += fi.Size()
		}
	}
	return total
}

// GetContextDBPath returns the context/docs/memory database path.
func GetContextDBPath() string {
	return contextDBPath()
}

func walPathFor(dbPath string) string {
	return dbPath + "-wal"
}
