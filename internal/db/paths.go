package db

import (
	"os"
	"path/filepath"
)

const (
	indexFile   = "index.db"
	contextFile = "context.db"
	usageFile   = "usage.db"
)

func cacheDir() string {
	if p := os.Getenv("DB_PATH"); p != "" {
		return filepath.Dir(p)
	}
	home := os.Getenv("HOME")
	if home == "" {
		return ".astcache"
	}
	return filepath.Join(home, ".astcache")
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

// GetContextDBPath returns the context/docs/memory database path.
func GetContextDBPath() string {
	return contextDBPath()
}

func walPathFor(dbPath string) string {
	return dbPath + "-wal"
}
