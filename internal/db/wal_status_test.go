package db

import (
	"testing"
	"time"
)

func TestWALMaintenanceLifecycle(t *testing.T) {
	endWALMaintenance()
	walStatusMu.Lock()
	walStatus.active = false
	walStatusMu.Unlock()

	beginWALMaintenance("test")
	if !WALMaintenanceActive() {
		t.Fatal("expected active after begin")
	}
	s := GetWALSnapshot()
	if s.Reason != "test" || s.Phase != WALPhasePausing {
		t.Fatalf("snapshot=%+v", s)
	}
	if s.WalStartBytes < 0 {
		t.Fatal("expected non-negative wal start bytes")
	}

	setWALPhase(WALPhaseCheckpoint, "RESTART")
	recordWALCheckpointResult(1, 100, 0, nil)
	s = GetWALSnapshot()
	if s.Phase != WALPhaseCheckpoint || s.Mode != "RESTART" || s.LastBusy != 1 {
		t.Fatalf("checkpoint snapshot=%+v", s)
	}

	endWALMaintenance()
	if WALMaintenanceActive() {
		t.Fatal("expected inactive after end")
	}
}

func TestRunManualWALCheckpointConflict(t *testing.T) {
	beginWALMaintenance("test")
	defer endWALMaintenance()

	started, errMsg := RunManualWALCheckpoint()
	if started || errMsg == "" {
		t.Fatalf("started=%v errMsg=%q want conflict", started, errMsg)
	}
	time.Sleep(10 * time.Millisecond)
}
