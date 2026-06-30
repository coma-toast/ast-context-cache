package components

import (
	"fmt"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func (h IndexHealth) ShowWALMaintenanceBanner() bool {
	return h.WALMaintenanceActive
}

func (h IndexHealth) WALMaintenanceHeadline() string {
	return "Compacting database WAL"
}

func (h IndexHealth) WALMaintenanceDetail() string {
	elapsed := walMaintenanceElapsed(h.WALMaintenanceStarted)
	switch h.WALMaintenancePhase {
	case string(db.WALPhasePausing):
		if h.WALInFlight > 0 {
			return fmt.Sprintf("Pausing embed workers… (%d in flight) · %s", h.WALInFlight, elapsed)
		}
		return fmt.Sprintf("Pausing embed workers… · %s", elapsed)
	case string(db.WALPhaseDraining):
		return fmt.Sprintf("Flushing write buffers… · %s", elapsed)
	case string(db.WALPhaseCheckpoint):
		return walCheckpointDetail(h, elapsed)
	case string(db.WALPhaseRestoring):
		return fmt.Sprintf("Restoring embed workers… · %s", elapsed)
	default:
		return fmt.Sprintf("Checkpoint in progress… · %s", elapsed)
	}
}

func walCheckpointDetail(h IndexHealth, elapsed string) string {
	start := db.FormatFileSize(h.WALWalStartBytes)
	cur := db.FormatFileSize(h.WALWalCurrentBytes)
	mode := h.WALMaintenanceMode
	if mode == "" {
		mode = "TRUNCATE"
	}
	if h.WALLastBusy == 1 {
		return fmt.Sprintf("Waiting for DB readers (busy) · %s → %s · streak %d · %s", start, cur, h.WALBusyStreak, elapsed)
	}
	if h.WALWalStartBytes > 0 && h.WALWalCurrentBytes < h.WALWalStartBytes {
		return fmt.Sprintf("Checkpoint %s · %s → %s · %s", mode, start, cur, elapsed)
	}
	return fmt.Sprintf("Checkpoint %s · %s · %s", mode, cur, elapsed)
}

func (h IndexHealth) WALMaintenanceProgressPct() float64 {
	if !h.WALMaintenanceActive || h.WALWalStartBytes <= 0 {
		return 0
	}
	if h.WALLastBusy == 1 && h.WALWalCurrentBytes >= h.WALWalStartBytes {
		return 0
	}
	shrink := 1 - float64(h.WALWalCurrentBytes)/float64(h.WALWalStartBytes)
	if shrink < 0 {
		shrink = 0
	}
	if shrink > 1 {
		shrink = 1
	}
	return shrink * 100
}

func (h IndexHealth) WALMaintenanceIndeterminate() bool {
	return h.WALMaintenanceActive && h.WALLastBusy == 1 && h.WALMaintenanceProgressPct() < 1
}

func (h IndexHealth) WALCheckpointButtonDisabled() bool {
	return h.WALMaintenanceActive || h.ShowStartupBanner()
}

func walMaintenanceElapsed(start time.Time) string {
	if start.IsZero() {
		return "0s"
	}
	d := time.Since(start).Round(time.Second)
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	m := int(d.Minutes())
	s := int(d.Seconds()) % 60
	return fmt.Sprintf("%dm%02ds", m, s)
}

func (h IndexHealth) WorkerWalBadgeText(target, effective int) string {
	if h.WALMaintenanceActive {
		return "checkpointing"
	}
	if effective < target && target > 0 {
		return fmt.Sprintf("WAL %d/%d", effective, target)
	}
	return ""
}

func (h IndexHealth) WorkerWalBadgeTitle(target, effective int) string {
	if h.WALMaintenanceActive {
		return h.WALMaintenanceHeadline() + " — " + h.WALMaintenanceDetail()
	}
	if effective >= target {
		return ""
	}
	return fmt.Sprintf("SQLite WAL throttled: %d of %d worker goroutines running", effective, target)
}
