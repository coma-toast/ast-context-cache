package components

import (
	"strings"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestWALMaintenanceDetailPhases(t *testing.T) {
	start := time.Now().Add(-45 * time.Second)
	h := IndexHealth{
		WALMaintenanceActive:  true,
		WALMaintenancePhase:   string(db.WALPhasePausing),
		WALMaintenanceStarted: start,
		WALInFlight:           2,
	}
	d := h.WALMaintenanceDetail()
	if !strings.Contains(d, "Pausing embed workers") || !strings.Contains(d, "2 in flight") {
		t.Fatalf("pausing detail=%q", d)
	}

	h.WALMaintenancePhase = string(db.WALPhaseCheckpoint)
	h.WALMaintenanceMode = "RESTART"
	h.WALLastBusy = 1
	h.WALWalStartBytes = 512 * 1024 * 1024
	h.WALWalCurrentBytes = 512 * 1024 * 1024
	h.WALBusyStreak = 3
	d = h.WALMaintenanceDetail()
	if !strings.Contains(d, "busy") || !strings.Contains(d, "streak 3") {
		t.Fatalf("busy detail=%q", d)
	}

	h.WALMaintenanceStarted = time.Now().Add(-3 * time.Minute)
	d = h.WALMaintenanceDetail()
	if !strings.Contains(d, "deferring until readers idle") {
		t.Fatalf("deferred detail=%q", d)
	}
}

func TestWALMaintenanceProgressPct(t *testing.T) {
	h := IndexHealth{
		WALMaintenanceActive: true,
		WALWalStartBytes:     1000,
		WALWalCurrentBytes:   250,
	}
	got := h.WALMaintenanceProgressPct()
	if got < 74 || got > 76 {
		t.Fatalf("progress=%v want ~75", got)
	}
	h.WALLastBusy = 1
	h.WALWalCurrentBytes = 1000
	if !h.WALMaintenanceIndeterminate() {
		t.Fatal("expected indeterminate when busy and no shrink")
	}
}

func TestDiskSizeLabelCompacting(t *testing.T) {
	h := IndexHealth{
		DiskSize:             "120 MB",
		WALMaintenanceActive: true,
		WALWalStartBytes:     512 * 1024 * 1024,
		WALWalCurrentBytes:   400 * 1024 * 1024,
	}
	got := h.DiskSizeLabel()
	if !strings.Contains(got, "→") {
		t.Fatalf("label=%q want shrink arrow", got)
	}
}
