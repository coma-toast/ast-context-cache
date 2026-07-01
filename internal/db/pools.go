package db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	_ "github.com/mattn/go-sqlite3"
)

var (
	// DB is the usage pool (queries, sessions, settings). Legacy name kept for callers.
	DB *sql.DB
	// IndexDB holds symbols, edges, vectors, embed_pending, summaries.
	IndexDB *sql.DB
	// ContextDB holds context notes, structured memory, docs, kv_repair_events.
	ContextDB *sql.DB
)

// Index returns the code index pool (read-only callers may use directly).
func Index() *sql.DB { return IndexDB }

// Context returns the context/docs pool.
func Context() *sql.DB { return ContextDB }

// Usage returns the usage/analytics pool (same as DB).
func Usage() *sql.DB { return DB }

// PoolsReady reports whether all database pools are open.
func PoolsReady() bool {
	return IndexDB != nil && ContextDB != nil && DB != nil
}

func openPool(path string) (*sql.DB, error) {
	if err := os.MkdirAll(cacheDir(), 0755); err != nil {
		return nil, err
	}
	dsn := path + "?_journal_mode=WAL&_busy_timeout=15000"
	conn, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, err
	}
	conn.SetMaxOpenConns(4)
	conn.SetMaxIdleConns(4)
	applyPragmas(conn)
	return conn, nil
}

func applyPragmas(conn *sql.DB) {
	conn.Exec(`PRAGMA journal_mode=WAL`)
	conn.Exec(`PRAGMA busy_timeout=15000`)
	conn.Exec(`PRAGMA synchronous=NORMAL`)
	conn.Exec(`PRAGMA cache_size=-32000`)
	conn.Exec(`PRAGMA wal_autocheckpoint=200`)
}

// Close closes all database pools (tests and shutdown).
func Close() {
	stopIndexWriter()
	for _, c := range []*sql.DB{IndexDB, ContextDB, DB} {
		if c != nil {
			c.Close()
		}
	}
	IndexDB, ContextDB, DB = nil, nil, nil
}

func statWalBytes(path string) int64 {
	fi, err := os.Stat(walPathFor(path))
	if err != nil {
		return 0
	}
	return fi.Size()
}

func walFileBytes() int64 {
	var total int64
	for _, p := range []string{indexDBPath(), usageDBPath(), contextDBPath()} {
		total += statWalBytes(p)
	}
	return total
}

// WalFileBytes returns combined on-disk WAL size across index, usage, and context databases.
func WalFileBytes() int64 {
	return walFileBytes()
}

// TotalDBBytes returns combined main-file size for index, usage, and context databases.
func TotalDBBytes() int64 {
	var total int64
	for _, p := range []string{indexDBPath(), usageDBPath(), contextDBPath()} {
		if fi, err := os.Stat(p); err == nil {
			total += fi.Size()
		}
	}
	return total
}

func dbLabel(path string) string {
	switch filepath.Base(path) {
	case indexFile:
		return "index"
	case contextFile:
		return "context"
	default:
		return "usage"
	}
}

func fmtOpenErr(which, path string, err error) error {
	return fmt.Errorf("open %s db %s: %w", which, path, err)
}
