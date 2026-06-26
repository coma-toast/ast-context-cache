package db

import (
	"testing"
)

func TestTrackCheckpointResult(t *testing.T) {
	walBusyStreak.Store(0)
	trackCheckpointResult(true, 1, 100)
	trackCheckpointResult(true, 1, 100)
	if got := walBusyStreak.Load(); got != 2 {
		t.Fatalf("busy streak=%d want 2", got)
	}
	trackCheckpointResult(true, 0, 100)
	if got := walBusyStreak.Load(); got != 0 {
		t.Fatalf("busy streak after success=%d want 0", got)
	}
}

func TestMaybeForceCheckpointThreshold(t *testing.T) {
	walBusyStreak.Store(0)
	if shouldForceCheckpoint(walForceBytes - 1) {
		t.Fatal("should not force below walForceBytes without busy streak")
	}
	walBusyStreak.Store(3)
	if !shouldForceCheckpoint(walTruncateBytes) {
		t.Fatal("should force when busy streak >= 3")
	}
	walBusyStreak.Store(0)
	if !shouldForceCheckpoint(walForceBytes) {
		t.Fatal("should force at walForceBytes")
	}
}
