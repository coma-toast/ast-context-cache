//go:build integration

package db

import (
	"database/sql"
	"os"
	"testing"
	"time"
)

func TestMaintainWALWithConcurrentReader(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				if IndexDB == nil {
					time.Sleep(10 * time.Millisecond)
					continue
				}
				rows, err := IndexDB.Query("SELECT COUNT(*) FROM symbols")
				if err == nil {
					rows.Close()
				}
				time.Sleep(50 * time.Millisecond)
			}
		}
	}()

	start := time.Now()
	busy, _, _, err := maintainWAL("integration", false)
	elapsed := time.Since(start)
	close(stop)

	if elapsed > checkpointMaxElapsed+5*time.Second {
		t.Fatalf("maintainWAL hung %v", elapsed)
	}
	if err != nil && busy == 0 {
		t.Logf("maintainWAL err=%v busy=%d elapsed=%v", err, busy, elapsed)
	}
}

func TestMaintainWALTruncateAfterQuiesce(t *testing.T) {
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
	if _, err := IndexDB.Exec(`CREATE TABLE IF NOT EXISTS int_wal (id INTEGER PRIMARY KEY, blob TEXT)`); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		IndexDB.Exec(`INSERT INTO int_wal (blob) VALUES (?)`, string(make([]byte, 8192)))
	}

	reader, _ := sql.Open("sqlite3", idxPath+"?_journal_mode=WAL")
	reader.SetMaxOpenConns(1)
	tx, _ := reader.Begin()
	tx.Exec(`SELECT 1`)

	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}
	_ = tx.Rollback()
	reader.Close()

	busy, _, _, err := checkpointFile(idxPath, "TRUNCATE")
	if err != nil || busy != 0 {
		t.Fatalf("TRUNCATE after quiesce busy=%d err=%v", busy, err)
	}
	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}
}
