package sys

import (
	"database/sql"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func openInMemorySQLite(t *testing.T) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open :memory: db: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

func TestStoreDiskHistoryUpsertAndPrune(t *testing.T) {
	conn := openInMemorySQLite(t)

	now := time.Now().UTC().Truncate(time.Second)
	disks := []DiskInfo{
		{
			Device: "/dev/nvme0n1", Model: "Samsung 980", Protocol: "nvme",
			SmartSource: "smartctl", SmartStatus: "PASSED",
			WearUsedPct: 42, SparePct: 58, SSDLifeLeft: 58,
			TemperatureC: 38.5, PowerOnHours: 1200,
			WarningLevel: WarningOK,
		},
		{
			Device: "/dev/sda", Model: "WD Red", Protocol: "sata",
			SmartSource: "smartctl", SmartStatus: "Verified",
			TemperatureC: 41, ReallocatedSectors: 3,
			WarningLevel: WarningWarning, WarningMsg: "3 reallocated sector(s)",
			ZFSPool: "tank", ZFSHealth: "online", ZFSCapacityPct: 62,
		},
	}
	if err := storeDiskHistory(conn, now, disks); err != nil {
		t.Fatalf("store: %v", err)
	}

	var count int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM disk_history WHERE ts = ?`, now.Format(diskHistoryTSFormat)).Scan(&count); err != nil {
		t.Fatalf("count: %v", err)
	}
	if count != 2 {
		t.Fatalf("row count = %d, want 2", count)
	}

	var model, warnLevel, warnMsg, zfsHealth string
	err := conn.QueryRow(`SELECT model, warning_level, warning_msg, zfs_health FROM disk_history WHERE device = ? AND ts = ?`,
		"/dev/sda", now.Format(diskHistoryTSFormat)).Scan(&model, &warnLevel, &warnMsg, &zfsHealth)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if model != "WD Red" || warnLevel != WarningWarning || warnMsg != "3 reallocated sector(s)" || zfsHealth != "online" {
		t.Fatalf("row = (%q, %q, %q, %q)", model, warnLevel, warnMsg, zfsHealth)
	}

	// Second sample at the same second replaces, not duplicates.
	if err := storeDiskHistory(conn, now, disks); err != nil {
		t.Fatalf("store again: %v", err)
	}
	if err := conn.QueryRow(`SELECT COUNT(*) FROM disk_history WHERE ts = ?`, now.Format(diskHistoryTSFormat)).Scan(&count); err != nil {
		t.Fatalf("count again: %v", err)
	}
	if count != 2 {
		t.Fatalf("row count after re-sample = %d, want 2", count)
	}

	// A row older than the retention window is pruned on the next store.
	old := now.AddDate(0, 0, -DiskHistoryRetentionDays-5)
	if _, err := conn.Exec(`INSERT INTO disk_history (ts, device, warning_level) VALUES (?, '/dev/old', 'ok')`, old.Format(diskHistoryTSFormat)); err != nil {
		t.Fatalf("insert old row: %v", err)
	}
	if err := storeDiskHistory(conn, now.Add(2*time.Second), disks); err != nil {
		t.Fatalf("store after prune: %v", err)
	}
	if err := conn.QueryRow(`SELECT COUNT(*) FROM disk_history WHERE device = '/dev/old'`).Scan(&count); err != nil {
		t.Fatalf("count old: %v", err)
	}
	if count != 0 {
		t.Fatalf("old rows = %d, want 0 (pruned)", count)
	}
}

func TestSampleDisksSnapshot(t *testing.T) {
	oldCollector := diskSampleCollector
	t.Cleanup(func() { diskSampleCollector = oldCollector })

	conn := openInMemorySQLite(t)
	oldDB := db.DB
	t.Cleanup(func() { db.DB = oldDB })
	db.DB = conn

	diskSampleCollector = func() ([]DiskInfo, string) {
		d1 := baseDisk()
		d1.Device = "/dev/nvme0n1"
		d1.WearUsedPct = 95
		d1.TemperatureC = 42
		d2 := baseDisk()
		d2.Device = "/dev/sda"
		d2.SmartStatus = "Verified"
		d2.TemperatureC = 30
		return []DiskInfo{d1, d2}, ""
	}

	sampleDisks()

	snap := DiskSamplerStatus()
	if !snap.LastOK {
		t.Fatalf("LastOK = false, err = %q", snap.LastError)
	}
	if snap.LastError != "" {
		t.Fatalf("LastError = %q, want empty", snap.LastError)
	}
	if snap.LastSampleAt.IsZero() {
		t.Fatal("LastSampleAt is zero")
	}
	if len(snap.Disks) != 2 {
		t.Fatalf("len(Disks) = %d, want 2", len(snap.Disks))
	}
	if snap.CriticalCount != 1 || snap.WarningCount != 0 {
		t.Fatalf("counts = (%d warning, %d critical), want (0, 1)", snap.WarningCount, snap.CriticalCount)
	}
	if snap.Disks[0].WarningLevel != WarningCritical {
		t.Fatalf("disk 0 level = %q, want critical", snap.Disks[0].WarningLevel)
	}

	var count int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM disk_history`).Scan(&count); err != nil {
		t.Fatalf("count: %v", err)
	}
	if count != 2 {
		t.Fatalf("history rows = %d, want 2", count)
	}
}

func TestSampleDisksCollectorNote(t *testing.T) {
	oldCollector := diskSampleCollector
	t.Cleanup(func() { diskSampleCollector = oldCollector })

	oldDB := db.DB
	t.Cleanup(func() { db.DB = oldDB })
	db.DB = nil

	diskSampleCollector = func() ([]DiskInfo, string) {
		return nil, "smartctl not found — install smartmontools (e.g. apt install smartmontools)"
	}

	sampleDisks()

	snap := DiskSamplerStatus()
	if !snap.LastOK {
		t.Fatalf("LastOK = false, err = %q", snap.LastError)
	}
	wantNote := "smartctl not found — install smartmontools (e.g. apt install smartmontools)"
	if snap.LastError != wantNote {
		t.Fatalf("LastError = %q, want %q", snap.LastError, wantNote)
	}
	if len(snap.Disks) != 0 {
		t.Fatalf("Disks = %v, want empty", snap.Disks)
	}
}

func TestStartStopDiskSampler(t *testing.T) {
	oldCollector := diskSampleCollector
	t.Cleanup(func() { diskSampleCollector = oldCollector })

	diskSampleCollector = func() ([]DiskInfo, string) {
		return nil, "test note"
	}

	// Reset shared package state so the test does not observe samples from
	// an earlier test in the package.
	StopDiskSampler()
	diskSamplerMu.Lock()
	diskSamplerSnap = DiskSamplerSnapshot{}
	diskSamplerMu.Unlock()

	StartDiskSampler(time.Hour)

	// The first sample runs immediately in the goroutine; wait for it.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		snap := DiskSamplerStatus()
		if !snap.LastSampleAt.IsZero() && snap.LastError == "test note" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	snap := DiskSamplerStatus()
	if !snap.Running {
		t.Fatal("Running = false while started")
	}
	if snap.LastError != "test note" {
		t.Fatalf("LastError = %q, want the collector note", snap.LastError)
	}

	StartDiskSampler(time.Hour) // must be a no-op, not a second loop
	StopDiskSampler()

	snap = DiskSamplerStatus()
	if snap.Running {
		t.Fatal("Running = true after stop")
	}
	if snap.LastError != "test note" {
		t.Fatalf("LastError after stop = %q, want preserved state", snap.LastError)
	}
}
