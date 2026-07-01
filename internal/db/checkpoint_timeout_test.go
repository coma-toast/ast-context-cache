package db

import (
	"os"
	"testing"
	"time"
)

func TestCheckpointCompletesWithinTimeout(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()

	idxPath := indexDBPath()
	growIndexWal(t)

	start := time.Now()
	_, _, _, err := checkpointFile(idxPath, "PASSIVE")
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > checkpointOpTimeout+5*time.Second {
		t.Fatalf("PASSIVE checkpoint took %v", elapsed)
	}

	start = time.Now()
	_, _, _, err = checkpointFile(idxPath, "TRUNCATE")
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > checkpointOpTimeout+5*time.Second {
		t.Fatalf("TRUNCATE checkpoint took %v", elapsed)
	}
}
