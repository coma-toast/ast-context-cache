package db

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestRunDataDirMoveRecreatesMissingSourceInsteadOfCopying simulates a USB drive that
// disconnected and came back with index.db missing while the pool handle stayed open
// (non-nil). The move must not try to VACUUM INTO from that dead source — which would
// have SQLite silently recreate an empty file there first — and instead start a fresh
// empty database at the target and report it as recreated, while still copying the two
// databases whose source files are intact.
func TestRunDataDirMoveRecreatesMissingSourceInsteadOfCopying(t *testing.T) {
	prevHome, prevDBPath := os.Getenv("HOME"), os.Getenv("DB_PATH")
	prevIndex, prevContext, prevUsage := IndexDB, ContextDB, DB
	t.Cleanup(func() {
		os.Setenv("HOME", prevHome)
		if prevDBPath == "" {
			os.Unsetenv("DB_PATH")
		} else {
			os.Setenv("DB_PATH", prevDBPath)
		}
		IndexDB, ContextDB, DB = prevIndex, prevContext, prevUsage
	})
	os.Setenv("HOME", t.TempDir())
	os.Unsetenv("DB_PATH")

	var err error
	if IndexDB, err = openPool(indexDBPath()); err != nil {
		t.Fatal(err)
	}
	if ContextDB, err = openPool(contextDBPath()); err != nil {
		t.Fatal(err)
	}
	if DB, err = openPool(usageDBPath()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, c := range []*sql.DB{IndexDB, ContextDB, DB} {
			if c != nil {
				c.Close()
			}
		}
	})

	if _, err := IndexDB.Exec(`CREATE TABLE t (id INTEGER)`); err != nil {
		t.Fatal(err)
	}
	// Delete the backing file out from under the still-open pool.
	os.Remove(indexDBPath())
	os.Remove(indexDBPath() + "-wal")
	os.Remove(indexDBPath() + "-shm")

	dataDirMoveMu.Lock()
	dataDirMove = DataDirMoveSnapshot{}
	dataDirMoveMu.Unlock()

	target := t.TempDir()
	started, errMsg := StartDataDirMove(target)
	if !started {
		t.Fatalf("move did not start: %s", errMsg)
	}

	deadline := time.Now().Add(5 * time.Second)
	var snap DataDirMoveSnapshot
	for time.Now().Before(deadline) {
		snap = GetDataDirMoveSnapshot()
		if !snap.Active {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if snap.Active {
		t.Fatal("move did not finish in time")
	}
	if snap.Error != "" {
		t.Fatalf("move failed: %s", snap.Error)
	}
	if len(snap.Recreated) != 1 || snap.Recreated[0] != indexFile {
		t.Fatalf("expected only %s recreated, got %v", indexFile, snap.Recreated)
	}
	if _, err := os.Stat(filepath.Join(target, indexFile)); err != nil {
		t.Fatalf("expected a fresh %s at target: %v", indexFile, err)
	}
	if _, err := os.Stat(filepath.Join(target, contextFile)); err != nil {
		t.Fatalf("expected copied %s at target: %v", contextFile, err)
	}
	if _, err := os.Stat(filepath.Join(target, usageFile)); err != nil {
		t.Fatalf("expected copied %s at target: %v", usageFile, err)
	}
}

// TestRunDataDirMoveRecreatesWhenPoolIsNil simulates IndexDB being nil at move time —
// e.g. WAL maintenance quiesced it (quiesceIndexPool) and restoreIndexPool couldn't
// reopen it because the drive holding it went missing. The move must not fail outright
// ("database not open"); it should start a fresh empty database at the target for that
// file, same as a missing source file, while still copying the other two.
func TestRunDataDirMoveRecreatesWhenPoolIsNil(t *testing.T) {
	prevHome, prevDBPath := os.Getenv("HOME"), os.Getenv("DB_PATH")
	prevIndex, prevContext, prevUsage := IndexDB, ContextDB, DB
	t.Cleanup(func() {
		os.Setenv("HOME", prevHome)
		if prevDBPath == "" {
			os.Unsetenv("DB_PATH")
		} else {
			os.Setenv("DB_PATH", prevDBPath)
		}
		IndexDB, ContextDB, DB = prevIndex, prevContext, prevUsage
	})
	os.Setenv("HOME", t.TempDir())
	os.Unsetenv("DB_PATH")

	var err error
	if ContextDB, err = openPool(contextDBPath()); err != nil {
		t.Fatal(err)
	}
	if DB, err = openPool(usageDBPath()); err != nil {
		t.Fatal(err)
	}
	IndexDB = nil
	t.Cleanup(func() {
		for _, c := range []*sql.DB{ContextDB, DB} {
			if c != nil {
				c.Close()
			}
		}
	})

	dataDirMoveMu.Lock()
	dataDirMove = DataDirMoveSnapshot{}
	dataDirMoveMu.Unlock()

	target := t.TempDir()
	started, errMsg := StartDataDirMove(target)
	if !started {
		t.Fatalf("move did not start: %s", errMsg)
	}

	deadline := time.Now().Add(5 * time.Second)
	var snap DataDirMoveSnapshot
	for time.Now().Before(deadline) {
		snap = GetDataDirMoveSnapshot()
		if !snap.Active {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if snap.Active {
		t.Fatal("move did not finish in time")
	}
	if snap.Error != "" {
		t.Fatalf("move failed: %s", snap.Error)
	}
	if len(snap.Recreated) != 1 || snap.Recreated[0] != indexFile {
		t.Fatalf("expected only %s recreated, got %v", indexFile, snap.Recreated)
	}
	if _, err := os.Stat(filepath.Join(target, indexFile)); err != nil {
		t.Fatalf("expected a fresh %s at target: %v", indexFile, err)
	}
	if _, err := os.Stat(filepath.Join(target, contextFile)); err != nil {
		t.Fatalf("expected copied %s at target: %v", contextFile, err)
	}
	if _, err := os.Stat(filepath.Join(target, usageFile)); err != nil {
		t.Fatalf("expected copied %s at target: %v", usageFile, err)
	}
}
