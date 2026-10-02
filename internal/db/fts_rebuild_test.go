package db

import (
	"context"
	"database/sql"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/mattn/go-sqlite3"
)

// endlessStmt runs until interrupted — a stand-in for a rebuild on a huge index.
// Each step calls fts_test_stepping(), which signals ftsTestStepping and then sleeps
// briefly, so an interrupted rebuild takes a moment to exit — long enough for a test
// to catch Close() returning before it has.
const endlessStmt = `WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE fts_test_stepping()) SELECT count(*) FROM c`

var (
	steppingDriverOnce sync.Once
	ftsTestStepping    = make(chan struct{}, 1)
)

// execStmt stands in for rebuildFTSTable: it runs stmt for every table.
func execStmt(stmt string) func(execer, string) error {
	return func(e execer, _ string) error {
		_, err := e.Exec(stmt)
		return err
	}
}

// initWithFTSRebuild runs Init with rebuild in place of rebuildFTSTable.
func initWithFTSRebuild(t *testing.T, rebuild func(execer, string) error) {
	t.Helper()
	prevHome := os.Getenv("HOME")
	os.Setenv("HOME", t.TempDir())
	prevRebuild := ftsRebuildTable
	ftsRebuildTable = rebuild
	t.Cleanup(func() {
		ftsRebuildTable = prevRebuild
		os.Setenv("HOME", prevHome)
	})
	if err := Init(); err != nil {
		t.Fatal(err)
	}
}

// startEndlessFTSRebuild starts endlessStmt as the tracked FTS rebuild and returns
// once it is stepping: go-sqlite3 drops a cancel that lands before then, which would
// leave the test waiting forever. It runs on its own pool so it can register
// fts_test_stepping(); the caller's Close() must still cancel it.
func startEndlessFTSRebuild(t *testing.T) {
	t.Helper()
	steppingDriverOnce.Do(func() {
		sql.Register("sqlite3_fts_test_stepping", &sqlite3.SQLiteDriver{
			ConnectHook: func(c *sqlite3.SQLiteConn) error {
				return c.RegisterFunc("fts_test_stepping", func() bool {
					select {
					case ftsTestStepping <- struct{}{}:
					default:
					}
					time.Sleep(20 * time.Millisecond)
					return true
				}, false)
			},
		})
	})
	pool, err := sql.Open("sqlite3_fts_test_stepping", indexDBPath())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { pool.Close() })
	select {
	case <-ftsTestStepping:
	default:
	}
	ftsRebuildTable = execStmt(endlessStmt)
	startFTSRebuild(pool)
	select {
	case <-ftsTestStepping:
	case <-time.After(10 * time.Second):
		t.Fatal("endless FTS rebuild never started stepping")
	}
}

func TestCloseCancelsInitFTSRebuild(t *testing.T) {
	initWithFTSRebuild(t, execStmt(`SELECT 1`))
	startEndlessFTSRebuild(t)

	closed := make(chan struct{})
	go func() {
		Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not interrupt the FTS rebuild")
	}
	if ftsRebuildRunning() {
		t.Fatal("FTS rebuild still running after Close returned")
	}
}

func TestQuiesceWaitsForInitFTSRebuild(t *testing.T) {
	initWithFTSRebuild(t, execStmt(`SELECT 1`))
	defer func() {
		indexReadGate.Store(false)
		Close()
	}()
	waitFTSRebuild()
	if _, err := IndexDB.Exec(`CREATE TABLE fts_probe (n INTEGER)`); err != nil {
		t.Fatal(err)
	}

	// Hold the write lock from another connection so the rebuild is pinned
	// mid-statement (busy-waiting) until the test lets it go.
	locker, err := sql.Open("sqlite3", indexDBPath()+"?_busy_timeout=15000")
	if err != nil {
		t.Fatal(err)
	}
	defer locker.Close()
	lockConn, err := locker.Conn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer lockConn.Close()
	if _, err := lockConn.ExecContext(context.Background(), `BEGIN IMMEDIATE`); err != nil {
		t.Fatal(err)
	}

	// Loops once it has the lock, so a cancel (rather than a wait) would interrupt it.
	ftsRebuildTable = execStmt(`INSERT INTO fts_probe SELECT count(*) FROM (WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c LIMIT 200000) SELECT x FROM c)`)
	startFTSRebuild(IndexDB)
	if !ftsRebuildRunning() {
		t.Fatal("expected FTS rebuild to be blocked on the write lock")
	}

	quiesced := make(chan error, 1)
	go func() { quiesced <- quiesceIndexPool() }()
	select {
	case err := <-quiesced:
		t.Fatalf("quiesceIndexPool returned (err=%v) while the FTS rebuild was still running", err)
	case <-time.After(200 * time.Millisecond):
	}

	if _, err := lockConn.ExecContext(context.Background(), `COMMIT`); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-quiesced:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("quiesceIndexPool did not return after the FTS rebuild finished")
	}
	if ftsRebuildRunning() {
		t.Fatal("FTS rebuild still running after quiesceIndexPool returned")
	}

	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := IndexDB.QueryRow(`SELECT COUNT(*) FROM fts_probe`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != len(symbolFTSTables) {
		t.Fatalf("fts_probe rows=%d, want %d (quiesce should let the rebuild finish, not cancel it)", n, len(symbolFTSTables))
	}
}

func TestMaintainWALSkipsQuiesceDuringFTSRebuild(t *testing.T) {
	initWithFTSRebuild(t, execStmt(`SELECT 1`))
	defer Close()
	startEndlessFTSRebuild(t)

	busy, _, _, err := maintainWAL("test", true)
	if err != nil || busy != 1 {
		t.Fatalf("maintainWAL busy=%d err=%v, want busy=1 err=nil", busy, err)
	}
	if IndexDB == nil || IndexReadQuiesced() {
		t.Fatal("maintainWAL quiesced the index pool while the FTS rebuild was running")
	}
	if got := GetWALSnapshot().SkipReason; got != "fts_rebuild" {
		t.Fatalf("skip reason=%q, want fts_rebuild", got)
	}
}
