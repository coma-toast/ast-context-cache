package db

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"
)

func TestIndexWalBytesNotCombined(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	idx := filepath.Join(dir, ".astcache", "index.db")
	if _, err := IndexDB.Exec(`CREATE TABLE IF NOT EXISTS wal_test (id INTEGER PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	if _, err := IndexDB.Exec(`INSERT INTO wal_test DEFAULT VALUES`); err != nil {
		t.Fatal(err)
	}
	_, _, _, _ = checkpointFile(idx, "PASSIVE")

	total := WalFileBytes()
	indexOnly := IndexWalBytes()
	if indexOnly <= 0 {
		t.Fatalf("index wal=%d want >0", indexOnly)
	}
	if indexOnly > total {
		t.Fatalf("index wal=%d total=%d", indexOnly, total)
	}
}

func TestIndexReadGateBlocksDuringQuiesce(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		indexReadGate.Store(false)
		_ = restoreIndexPool()
		Close()
	}()

	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}
	if !IndexReadQuiesced() {
		t.Fatal("expected gate during quiesce")
	}
	if IndexDB != nil {
		t.Fatal("expected nil IndexDB during quiesce")
	}
	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}
	if IndexReadQuiesced() {
		t.Fatal("gate should clear after restore")
	}
	if IndexDB == nil {
		t.Fatal("expected IndexDB after restore")
	}
}

func TestRestoreIndexPoolReopensWriter(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		indexReadGate.Store(false)
		Close()
	}()

	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}
	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}
	err := IndexWrite(func(tx *sql.Tx) error {
		_, e := tx.Exec(`CREATE TABLE IF NOT EXISTS writer_test (id INTEGER PRIMARY KEY)`)
		return e
	})
	if err != nil {
		t.Fatalf("IndexWrite after restore: %v", err)
	}
}

func TestIndexReaderDuringQuiesce(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		indexReadGate.Store(false)
		_ = restoreIndexPool()
		Close()
	}()

	if _, err := IndexReader(); err != nil {
		t.Fatalf("expected reader ok: %v", err)
	}
	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}
	if _, err := IndexReader(); err == nil {
		t.Fatal("expected error during quiesce")
	}
}

func growIndexWal(t *testing.T) {
	t.Helper()
	if _, err := IndexDB.Exec(`PRAGMA wal_autocheckpoint=0`); err != nil {
		t.Fatal(err)
	}
	if _, err := IndexDB.Exec(`CREATE TABLE IF NOT EXISTS wal_shrink (id INTEGER PRIMARY KEY, data TEXT)`); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 500; i++ {
		if _, err := IndexDB.Exec(`INSERT INTO wal_shrink (data) VALUES (?)`, string(make([]byte, 8192))); err != nil {
			t.Fatal(err)
		}
	}
}

func TestQuiesceIndexPoolTruncateShrinksWal(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		indexReadGate.Store(false)
		_ = restoreIndexPool()
		Close()
	}()

	idxPath := indexDBPath()
	growIndexWal(t)
	before := IndexWalBytes()

	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}

	busy, _, _, err := checkpointFile(idxPath, "TRUNCATE")
	if err != nil {
		_ = restoreIndexPool()
		t.Fatal(err)
	}
	if busy != 0 {
		_ = restoreIndexPool()
		t.Fatalf("TRUNCATE after quiesce busy=%d", busy)
	}
	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}

	after := IndexWalBytes()
	if after >= before && before > 4096 {
		t.Fatalf("wal before=%d after=%d want shrink", before, after)
	}
}
