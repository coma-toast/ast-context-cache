package db

import (
	"context"
	"database/sql"
	"sync"
	"time"
)

// ftsRebuildTable is what Init's background rebuild runs for each of
// symbolFTSTables. A var so tests can swap in something slower or cancellable.
var ftsRebuildTable = rebuildFTSTable

// ctxExecer runs every Exec under ctx, so cancelling ctx interrupts it.
type ctxExecer struct {
	db  *sql.DB
	ctx context.Context
}

func (e ctxExecer) Exec(query string, args ...any) (sql.Result, error) {
	return e.db.ExecContext(e.ctx, query, args...)
}

// Init's FTS rebuild runs in the background (it can take minutes on a large index),
// but it's tracked rather than fire-and-forget: sql.DB.Close only closes idle
// connections, so a rebuild already running on a busy one would keep writing
// index.db-wal/-shm after Close() or quiesceIndexPool() had "closed" the pool —
// overlapping a WAL TRUNCATE in production, and a t.TempDir() cleanup in tests.
var (
	ftsRebuildMu     sync.Mutex
	ftsRebuildCancel context.CancelFunc
	ftsRebuildDone   chan struct{}
)

// startFTSRebuild rebuilds symbolFTSTables on idx in the background. The goroutine
// gets the pool handle and rebuild func up front rather than reading package vars
// that Close(), quiesceIndexPool() and tests reassign while it runs.
func startFTSRebuild(idx *sql.DB) {
	cancelFTSRebuild()
	rebuild := ftsRebuildTable
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	ftsRebuildMu.Lock()
	ftsRebuildCancel, ftsRebuildDone = cancel, done
	ftsRebuildMu.Unlock()
	go func() {
		defer close(done)
		defer cancel()
		start := time.Now()
		for _, table := range symbolFTSTables {
			if err := rebuild(ctxExecer{idx, ctx}, table); err != nil {
				if ctx.Err() != nil {
					logger.Info("FTS startup rebuild interrupted", "elapsed", time.Since(start).Round(time.Millisecond))
					return
				}
				logger.Warn("FTS startup rebuild failed", "error", err)
			}
		}
		logger.Info("FTS startup rebuild finished", "duration", time.Since(start).Round(time.Millisecond))
	}()
}

func ftsRebuildState() (context.CancelFunc, chan struct{}) {
	ftsRebuildMu.Lock()
	defer ftsRebuildMu.Unlock()
	return ftsRebuildCancel, ftsRebuildDone
}

// waitFTSRebuild blocks until Init's background FTS rebuild (if any) has exited.
func waitFTSRebuild() {
	if _, done := ftsRebuildState(); done != nil {
		<-done
	}
}

// cancelFTSRebuild interrupts Init's background FTS rebuild (if any) and waits for it
// to exit. The interrupted statement rolls back; the next Init rebuilds again.
// go-sqlite3 cancels with a single sqlite3_interrupt, which SQLite drops if it lands
// before the statement has started stepping — in that narrow window this waits for
// the statement to finish instead.
func cancelFTSRebuild() {
	if cancel, done := ftsRebuildState(); done != nil {
		cancel()
		<-done
	}
}

func ftsRebuildRunning() bool {
	_, done := ftsRebuildState()
	if done == nil {
		return false
	}
	select {
	case <-done:
		return false
	default:
		return true
	}
}
