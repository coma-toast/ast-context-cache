package db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"

	_ "github.com/mattn/go-sqlite3"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	pragmaJournalModeWAL    = `PRAGMA journal_mode=WAL`
	pragmaBusyTimeout       = `PRAGMA busy_timeout=15000`
	pragmaSynchronousNormal = `PRAGMA synchronous=NORMAL`
	pragmaCacheSize         = `PRAGMA cache_size=-32000`
	pragmaWALAutocheckpoint = `PRAGMA wal_autocheckpoint=200`
	poolDSNParams           = "?_journal_mode=WAL&_busy_timeout=15000"
	handoffTxLockParam      = "&_txlock=immediate"
)

var (
	// DB is the usage pool (queries, sessions, settings). Legacy name kept for callers.
	DB *sql.DB
	// IndexDB holds symbols, edges, vectors, embed_pending, summaries.
	IndexDB *sql.DB
	// ContextDB holds context notes, structured memory, docs, kv_repair_events, and handoff trees.
	ContextDB *sql.DB
	// HandoffWriteDB is a second, one-connection pool on context.db whose transactions begin
	// IMMEDIATE. Every handoff, scratchpad, and claim write goes through it (see HandoffTx), so
	// they are linearized and a read-then-write never has to upgrade a DEFERRED transaction.
	HandoffWriteDB *sql.DB
)

// Index returns the code index pool (read-only callers may use directly).
func Index() *sql.DB { return IndexDB }

// Context returns the context/docs pool.
func Context() *sql.DB { return ContextDB }

// Usage returns the usage/analytics pool (same as DB).
func Usage() *sql.DB { return DB }

// poolsOpen mirrors whether all three pools are open. It's updated after every
// assignment to them, so PoolsReady can answer without reading the pool vars:
// the dashboard's live-refresh loop polls it from a goroutine started at package
// init (before main's Init), and in tests while each one's Init/Close reassigns
// the pools, which was a data race.
var poolsOpen atomic.Bool

func syncPoolsOpen() {
	poolsOpen.Store(IndexDB != nil && ContextDB != nil && DB != nil)
}

// PoolsReady reports whether all database pools are open.
func PoolsReady() bool {
	return poolsOpen.Load()
}

func openPool(path string) (*sql.DB, error) {
	return openPoolWith(path, poolDSNParams, 4)
}

// openHandoffWritePool opens HandoffWriteDB's single connection on the context database.
func openHandoffWritePool(path string) (*sql.DB, error) {
	return openPoolWith(path, poolDSNParams+handoffTxLockParam, 1)
}

func openPoolWith(path, params string, conns int) (*sql.DB, error) {
	if err := os.MkdirAll(cacheDir(), 0o755); err != nil {
		return nil, err
	}
	conn, err := sql.Open("sqlite3", path+params)
	if err != nil {
		return nil, err
	}
	conn.SetMaxOpenConns(conns)
	conn.SetMaxIdleConns(conns)
	applyPragmas(conn)
	return conn, nil
}

func applyPragmas(conn *sql.DB) {
	conn.Exec(pragmaJournalModeWAL)
	conn.Exec(pragmaBusyTimeout)
	conn.Exec(pragmaSynchronousNormal)
	conn.Exec(pragmaCacheSize)
	conn.Exec(pragmaWALAutocheckpoint)
}

// Close closes all database pools (tests and shutdown).
func Close() {
	poolsOpen.Store(false)
	stopWriteBatchers()
	stopIndexWriter()
	cancelFTSRebuild()
	for _, c := range []*sql.DB{IndexDB, HandoffWriteDB, ContextDB, DB} {
		if c != nil {
			c.Close()
		}
	}
	IndexDB, HandoffWriteDB, ContextDB, DB = nil, nil, nil, nil
	syncPoolsOpen()
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
	return errs.WrapMessage(fmt.Sprintf("failed to open %s db %s", which, path), err, "db", which, "path", path)
}
