package db

import (
	"log"
	"os"
	"sync"
	"syscall"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// driveMonitorInterval is how often the data directory's reachability is checked.
const driveMonitorInterval = 5 * time.Second

// driveMonitorMisses is how many consecutive failed checks (~10s at the interval
// above) are required before declaring the drive disconnected. A single stat failure
// can be a transient hiccup; this mirrors the same debounce internal/purge/sweep.go
// uses before treating a missing project directory as real.
const driveMonitorMisses = 2

// DriveDisconnectedSnapshot is a point-in-time view of the drive-monitor state for the
// dashboard, same shape as WALSnapshot.
type DriveDisconnectedSnapshot struct {
	Active     bool
	Path       string
	DetectedAt time.Time
}

var (
	driveMonitorMu  sync.RWMutex
	driveMonitor    DriveDisconnectedSnapshot
	driveMonitorDev uint64
	driveMonitorSet bool
	driveMisses     int
)

// GetDriveDisconnectedSnapshot returns the current drive-monitor state for the dashboard.
func GetDriveDisconnectedSnapshot() DriveDisconnectedSnapshot {
	driveMonitorMu.RLock()
	defer driveMonitorMu.RUnlock()
	return driveMonitor
}

// ResetDriveMonitorForTest clears all debounce/baseline state (tests only).
func ResetDriveMonitorForTest() {
	driveMonitorMu.Lock()
	driveMonitor = DriveDisconnectedSnapshot{}
	driveMonitorDev = 0
	driveMonitorSet = false
	driveMisses = 0
	driveMonitorMu.Unlock()
}

// recordDriveCheck is the debounce logic for one tick, pure enough to test without a
// ticker or a real filesystem: given whether the data directory was reachable this
// tick and (if so) which device it's on, it returns whether disconnection should now
// be declared. ok=false means the stat failed outright; dev is ignored when ok=false.
func recordDriveCheck(dev uint64, ok bool) bool {
	driveMonitorMu.Lock()
	defer driveMonitorMu.Unlock()
	switch {
	case !ok:
		driveMisses++
	case !driveMonitorSet:
		driveMonitorDev = dev
		driveMonitorSet = true
		driveMisses = 0
	case dev != driveMonitorDev:
		// Reachable, but no longer the same filesystem — e.g. an unmounted mount
		// point that still exists as an empty directory on the parent disk.
		driveMisses++
	default:
		driveMisses = 0
	}
	return driveMisses >= driveMonitorMisses
}

// StartDriveMonitor periodically checks that the configured data directory is still
// reachable and still the same filesystem it was when this process started, so a
// mid-session USB/external drive removal is caught within a few seconds instead of
// discovered only when a query fails or the process crashes.
//
// This cannot prevent every crash: SQLite's WAL mode always memory-maps its -shm
// index file, and touching a mapped page whose backing device just vanished raises
// SIGBUS at the OS level — a signal that arrives while control is inside SQLite's C
// code (via cgo) is fatal in Go and cannot be recovered from, no matter what checks
// run elsewhere. What this does do is shrink the window: once disconnection is
// detected, embed workers are paused and no new database work is started, so most
// activity stops before it can hit the dead device. There is no live-reconnect path —
// once tripped, this stays tripped until the process is restarted, matching how
// Init() already refuses to start cleanly if the configured directory is missing.
func StartDriveMonitor() {
	ticker := time.NewTicker(driveMonitorInterval)
	defer ticker.Stop()
	for range ticker.C {
		if GetDriveDisconnectedSnapshot().Active {
			return
		}
		dev, ok := statDev(GetDataDir())
		if recordDriveCheck(dev, ok) {
			declareDriveDisconnected()
			return
		}
	}
}

func statDev(path string) (uint64, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}
	return uint64(st.Dev), true
}

func declareDriveDisconnected() {
	path := GetDataDir()
	log.Printf("data directory %s is no longer reachable — the drive may have been disconnected; pausing embed workers, restart required to recover", path)
	checkpointAbort.Store(true)
	if BeforeForceCheckpoint != nil {
		BeforeForceCheckpoint()
	}
	driveMonitorMu.Lock()
	driveMonitor = DriveDisconnectedSnapshot{Active: true, Path: path, DetectedAt: time.Now()}
	driveMonitorMu.Unlock()
	realtime.Notify(realtime.IndexHealth)
}
