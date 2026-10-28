package sys

import (
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/logging"
)

// Sampling cadence and retention for the disk health history.
const (
	DiskSampleInterval       = 15 * time.Minute
	DiskHistoryRetentionDays = 30
	diskHistoryTSFormat      = "2006-01-02 15:04:05"
)

var diskSamplerLogger = logging.Tagged("disk_sampler")

const createDiskHistoryTable = `CREATE TABLE IF NOT EXISTS disk_history (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	ts TEXT NOT NULL,
	device TEXT NOT NULL,
	model TEXT,
	protocol TEXT,
	smart_source TEXT,
	smart_status TEXT,
	wear_pct INTEGER,
	spare_pct INTEGER,
	life_left_pct INTEGER,
	temp_c REAL,
	power_on_hours INTEGER,
	reallocated_sectors INTEGER,
	pending_sectors INTEGER,
	offline_uncorrectable INTEGER,
	media_errors INTEGER,
	zfs_pool TEXT,
	zfs_health TEXT,
	zfs_capacity_pct INTEGER,
	warning_level TEXT,
	warning_msg TEXT,
	UNIQUE(ts, device)
)`

const createDiskHistoryIndex = `CREATE INDEX IF NOT EXISTS idx_disk_history_device_ts ON disk_history (device, ts)`

const insertDiskHistoryQuery = `INSERT OR REPLACE INTO disk_history (
	ts, device, model, protocol, smart_source, smart_status,
	wear_pct, spare_pct, life_left_pct, temp_c, power_on_hours,
	reallocated_sectors, pending_sectors, offline_uncorrectable, media_errors,
	zfs_pool, zfs_health, zfs_capacity_pct, warning_level, warning_msg
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

const pruneDiskHistoryQuery = `DELETE FROM disk_history WHERE ts < ?`

// DiskSampleSummary is the per-disk view of the latest sample used by the
// Prometheus metrics and API endpoints.
type DiskSampleSummary struct {
	Device               string
	TemperatureC         float64
	WearUsedPct          int
	SparePct             int
	SSDLifeLeft          int
	ReallocatedSectors   int
	PendingSectors       int
	OfflineUncorrectable int
	WarningLevel         string
}

// DiskSamplerSnapshot is the latest sample state. LastOK is true when the
// cycle completed without internal error; LastError carries the collector's
// environment note (e.g. "smartctl not found") or a store failure.
type DiskSamplerSnapshot struct {
	Running       bool
	LastSampleAt  time.Time
	LastOK        bool
	LastError     string
	Disks         []DiskSampleSummary
	WarningCount  int
	CriticalCount int
}

// diskSampleCollector is replaced by tests.
var diskSampleCollector = func() ([]DiskInfo, string) {
	return DiskHealthInfo(db.GetDataDir())
}

var (
	diskSamplerMu   sync.Mutex
	diskSamplerStop chan struct{}
	diskSamplerSnap DiskSamplerSnapshot
)

// StartDiskSampler begins the periodic disk health sweep. It is a no-op if a
// sampler is already running. The first sample runs immediately.
func StartDiskSampler(interval time.Duration) {
	diskSamplerMu.Lock()
	if diskSamplerStop != nil {
		diskSamplerMu.Unlock()
		return
	}
	stop := make(chan struct{})
	diskSamplerStop = stop
	s := diskSamplerSnap
	s.Running = true
	diskSamplerSnap = s
	diskSamplerMu.Unlock()

	go diskSamplerLoop(interval, stop)
}

// StopDiskSampler stops the periodic sweep.
func StopDiskSampler() {
	diskSamplerMu.Lock()
	stop := diskSamplerStop
	diskSamplerStop = nil
	s := diskSamplerSnap
	s.Running = false
	diskSamplerSnap = s
	diskSamplerMu.Unlock()
	if stop != nil {
		close(stop)
	}
}

// DiskSamplerStatus returns a copy of the latest sample state.
func DiskSamplerStatus() DiskSamplerSnapshot {
	diskSamplerMu.Lock()
	defer diskSamplerMu.Unlock()
	s := diskSamplerSnap
	s.Disks = append([]DiskSampleSummary(nil), diskSamplerSnap.Disks...)
	return s
}

func diskSamplerLoop(interval time.Duration, stop <-chan struct{}) {
	sampleDisks()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			sampleDisks()
		}
	}
}

func sampleDisks() {
	now := time.Now().UTC()
	disks, note := diskSampleCollector()
	ApplyDiskWarnings(disks)

	s := DiskSamplerSnapshot{LastSampleAt: now, LastOK: true, LastError: note}

	if len(disks) > 0 {
		conn := db.Usage()
		if conn == nil {
			s.LastOK = false
			s.LastError = "database not ready"
		} else if err := storeDiskHistory(conn, now, disks); err != nil {
			s.LastOK = false
			s.LastError = "store disk history: " + err.Error()
			diskSamplerLogger.Warn("disk history sample not stored", "error", err)
		}
	}

	for _, d := range disks {
		s.Disks = append(s.Disks, DiskSampleSummary{
			Device:               d.Device,
			TemperatureC:         d.TemperatureC,
			WearUsedPct:          d.WearUsedPct,
			SparePct:             d.SparePct,
			SSDLifeLeft:          d.SSDLifeLeft,
			ReallocatedSectors:   d.ReallocatedSectors,
			PendingSectors:       d.PendingSectors,
			OfflineUncorrectable: d.OfflineUncorrectable,
			WarningLevel:         d.WarningLevel,
		})
		switch d.WarningLevel {
		case WarningWarning:
			s.WarningCount++
		case WarningCritical:
			s.CriticalCount++
		}
	}

	diskSamplerMu.Lock()
	running := diskSamplerStop != nil
	s.Running = running
	diskSamplerSnap = s
	diskSamplerMu.Unlock()

	if s.WarningCount > 0 || s.CriticalCount > 0 {
		diskSamplerLogger.Warn("disk health sample", "disks", len(disks), "warnings", s.WarningCount, "critical", s.CriticalCount, "note", note)
	} else {
		diskSamplerLogger.Debug("disk health sample", "disks", len(disks), "note", note)
	}
}

// storeDiskHistory upserts one sample row per device in the usage database
// and prunes rows older than the retention window.
func storeDiskHistory(conn *sql.DB, ts time.Time, disks []DiskInfo) error {
	if _, err := conn.Exec(createDiskHistoryTable); err != nil {
		return fmt.Errorf("create disk_history: %w", err)
	}
	if _, err := conn.Exec(createDiskHistoryIndex); err != nil {
		return fmt.Errorf("create disk_history index: %w", err)
	}
	tsStr := ts.Format(diskHistoryTSFormat)

	tx, err := conn.Begin()
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(insertDiskHistoryQuery)
	if err != nil {
		return fmt.Errorf("prepare insert: %w", err)
	}
	defer stmt.Close()

	for _, d := range disks {
		if _, err := stmt.Exec(
			tsStr, d.Device, d.Model, d.Protocol, d.SmartSource, d.SmartStatus,
			d.WearUsedPct, d.SparePct, d.SSDLifeLeft, d.TemperatureC, d.PowerOnHours,
			d.ReallocatedSectors, d.PendingSectors, d.OfflineUncorrectable, d.MediaErrors,
			d.ZFSPool, d.ZFSHealth, d.ZFSCapacityPct, d.WarningLevel, d.WarningMsg,
		); err != nil {
			return fmt.Errorf("insert %s: %w", d.Device, err)
		}
	}

	cutoff := ts.AddDate(0, 0, -DiskHistoryRetentionDays).Format(diskHistoryTSFormat)
	if _, err := tx.Exec(pruneDiskHistoryQuery, cutoff); err != nil {
		return fmt.Errorf("prune: %w", err)
	}

	return tx.Commit()
}
