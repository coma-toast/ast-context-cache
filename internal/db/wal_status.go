package db

import (
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// WALPhase is the current WAL maintenance step for dashboard display.
type WALPhase string

const (
	WALPhaseIdle        WALPhase = "idle"
	WALPhasePausing     WALPhase = "pausing"
	WALPhaseDraining    WALPhase = "draining"
	WALPhaseCheckpoint  WALPhase = "checkpoint"
	WALPhaseRestoring   WALPhase = "restoring"
)

// WALSnapshot is a point-in-time view of WAL maintenance for the dashboard.
type WALSnapshot struct {
	Active           bool
	Phase            WALPhase
	Mode             string
	Reason           string
	StartedAt        time.Time
	WalStartBytes    int64
	WalCurrentBytes  int64
	BusyStreak       int32
	InFlight         int64
	LastBusy         int
	LastFrames       int
	LastCheckpointed int
	LastError        string
	SkipReason       string
	NextAttemptAt    time.Time
}

// WALInFlightHook returns in-flight embed count during maintenance (set from main to avoid import cycles).
var WALInFlightHook func() int64

var (
	walStatusMu sync.RWMutex
	walStatus   struct {
		active           bool
		phase            WALPhase
		mode             string
		reason           string
		startedAt        time.Time
		walStartBytes    int64
		walCurrentBytes  int64
		inFlight         int64
		lastBusy         int
		lastFrames       int
		lastCheckpointed int
		lastError        string
		skipReason       string
		nextAttemptAt    time.Time
	}
	walHeartbeatStop chan struct{}
)

// WALMaintenanceActive reports whether a checkpoint cycle is in progress.
func WALMaintenanceActive() bool {
	walStatusMu.RLock()
	defer walStatusMu.RUnlock()
	return walStatus.active
}

// WALSnapshot returns current WAL maintenance state for dashboards.
func GetWALSnapshot() WALSnapshot {
	walStatusMu.RLock()
	defer walStatusMu.RUnlock()
	s := WALSnapshot{
		Active:           walStatus.active,
		Phase:            walStatus.phase,
		Mode:             walStatus.mode,
		Reason:           walStatus.reason,
		StartedAt:        walStatus.startedAt,
		WalStartBytes:    walStatus.walStartBytes,
		WalCurrentBytes:  walStatus.walCurrentBytes,
		BusyStreak:       walBusyStreak.Load(),
		InFlight:         walStatus.inFlight,
		LastBusy:         walStatus.lastBusy,
		LastFrames:       walStatus.lastFrames,
		LastCheckpointed: walStatus.lastCheckpointed,
		LastError:        walStatus.lastError,
		SkipReason:       walStatus.skipReason,
		NextAttemptAt:    walStatus.nextAttemptAt,
	}
	if s.Phase == "" {
		s.Phase = WALPhaseIdle
	}
	return s
}

func beginWALMaintenance(reason string) {
	walStatusMu.Lock()
	walStatus.active = true
	walStatus.phase = WALPhasePausing
	walStatus.mode = ""
	walStatus.reason = reason
	walStatus.startedAt = time.Now()
	walStatus.walStartBytes = IndexWalBytes()
	walStatus.walCurrentBytes = walStatus.walStartBytes
	walStatus.lastBusy = 0
	walStatus.lastFrames = 0
	walStatus.lastCheckpointed = 0
	walStatus.lastError = ""
	if WALInFlightHook != nil {
		walStatus.inFlight = WALInFlightHook()
	}
	stop := walHeartbeatStop
	walHeartbeatStop = make(chan struct{})
	walStatusMu.Unlock()
	if stop != nil {
		close(stop)
	}
	startWALHeartbeat(walHeartbeatStop)
	realtime.Notify(realtime.IndexHealth)
}

func setWALPhase(phase WALPhase, mode string) {
	walStatusMu.Lock()
	walStatus.phase = phase
	if mode != "" {
		walStatus.mode = mode
	}
	if WALInFlightHook != nil && (phase == WALPhasePausing || phase == WALPhaseDraining) {
		walStatus.inFlight = WALInFlightHook()
	}
	walStatusMu.Unlock()
	realtime.Notify(realtime.IndexHealth)
}

func recordWALCheckpointResult(busy, frames, checkpointed int, err error) {
	walStatusMu.Lock()
	walStatus.lastBusy = busy
	walStatus.lastFrames = frames
	walStatus.lastCheckpointed = checkpointed
	walStatus.walCurrentBytes = IndexWalBytes()
	if err != nil {
		walStatus.lastError = err.Error()
	}
	walStatusMu.Unlock()
	realtime.Notify(realtime.IndexHealth)
}

func endWALMaintenance() {
	walStatusMu.Lock()
	walStatus.active = false
	walStatus.phase = WALPhaseIdle
	walStatus.mode = ""
	walStatus.walCurrentBytes = IndexWalBytes()
	stop := walHeartbeatStop
	walHeartbeatStop = nil
	walStatusMu.Unlock()
	if stop != nil {
		close(stop)
	}
	realtime.Notify(realtime.IndexHealth)
}

func startWALHeartbeat(stop <-chan struct{}) {
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				walHeartbeatTick()
			}
		}
	}()
}

func walHeartbeatTick() {
	walStatusMu.Lock()
	if !walStatus.active {
		walStatusMu.Unlock()
		return
	}
	walStatus.walCurrentBytes = IndexWalBytes()
	if WALInFlightHook != nil && (walStatus.phase == WALPhasePausing || walStatus.phase == WALPhaseDraining) {
		walStatus.inFlight = WALInFlightHook()
	}
	walStatusMu.Unlock()
	realtime.Notify(realtime.IndexHealth)
}

// RunManualWALCheckpoint starts a manual WAL checkpoint in the background if not already running.
func RunManualWALCheckpoint() (started bool, errMsg string) {
	if WALMaintenanceActive() {
		return false, "WAL checkpoint already in progress"
	}
	go func() {
		maintainWAL("manual", true)
	}()
	return true, ""
}

func setWalSkipReason(reason string, until time.Time) {
	walStatusMu.Lock()
	walStatus.skipReason = reason
	walStatus.nextAttemptAt = until
	walStatusMu.Unlock()
}
