package db

import (
	"sync"
	"syscall"
	"time"
)

// Free-space thresholds for the data directory's volume. WAL backpressure (pressure.go)
// reacts to WAL size, not to how much room is left: a catch-up re-index has grown
// index.db-wal to ~4 GB, which a small or nearly full disk cannot absorb. Below
// diskLowFreeBytes embed workers (the main WAL writers) are capped; below
// diskCriticalFreeBytes they are paused so a WAL TRUNCATE can hand space back.
const (
	diskLowFreeBytes      = 5 << 30 // 5 GiB — room for one more multi-GB WAL burst
	diskCriticalFreeBytes = 1 << 30 // 1 GiB — SQLite writes start failing soon after
	diskLowWorkerCap      = 2       // same cap as walHighBytes in ThrottledEmbedWorkers

	DiskOK       = "ok"
	DiskLow      = "low"
	DiskCritical = "critical"
)

// DiskSpaceSnapshot is the last free-space sample for the data directory's volume.
type DiskSpaceSnapshot struct {
	Level     string
	FreeBytes uint64
	Path      string
	CheckedAt time.Time
}

var (
	// freeBytesFunc reports free bytes available to this user on dir's volume.
	// Swapped in tests via SetFreeBytesFuncForTest so nothing has to fill a disk.
	freeBytesFunc = statfsFreeBytes

	diskMu   sync.Mutex
	diskLast = DiskSpaceSnapshot{Level: DiskOK}
)

func statfsFreeBytes(dir string) (uint64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return uint64(st.Bavail) * uint64(st.Bsize), nil
}

func diskLevel(free uint64) string {
	switch {
	case free < diskCriticalFreeBytes:
		return DiskCritical
	case free < diskLowFreeBytes:
		return DiskLow
	default:
		return DiskOK
	}
}

// SampleDiskSpace measures free space on the data directory's volume, logs level
// changes, and returns the new snapshot. A failed measurement is treated as ok (never
// throttle on a guess) and keeps the previous level.
func SampleDiskSpace() DiskSpaceSnapshot {
	dir := GetDataDir()
	diskMu.Lock()
	probe := freeBytesFunc
	diskMu.Unlock()
	free, err := probe(dir)
	diskMu.Lock()
	defer diskMu.Unlock()
	if err != nil {
		return diskLast
	}
	prev := diskLast.Level
	diskLast = DiskSpaceSnapshot{Level: diskLevel(free), FreeBytes: free, Path: dir, CheckedAt: time.Now()}
	if diskLast.Level != prev {
		switch diskLast.Level {
		case DiskCritical:
			logger.Error("Disk space critical, pausing embed workers until space is freed", "free", FormatFileSize(int64(free)), "path", dir, "threshold", FormatFileSize(diskCriticalFreeBytes))
		case DiskLow:
			logger.Warn("Disk space low, capping embed workers", "free", FormatFileSize(int64(free)), "path", dir, "threshold", FormatFileSize(diskLowFreeBytes), "worker_cap", diskLowWorkerCap)
		default:
			logger.Info("Disk space recovered, lifting disk throttle", "free", FormatFileSize(int64(free)), "path", dir)
		}
	}
	return diskLast
}

// LastDiskSpace returns the most recent sample without measuring again.
func LastDiskSpace() DiskSpaceSnapshot {
	diskMu.Lock()
	defer diskMu.Unlock()
	return diskLast
}

// DiskSpaceCeiling samples free space and narrows an embed-worker ceiling
// (-1 = none, as returned by UpdateWalBackpressure) for low/critical disk.
func DiskSpaceCeiling(ceiling int) int {
	switch SampleDiskSpace().Level {
	case DiskCritical:
		return 0
	case DiskLow:
		if ceiling < 0 || ceiling > diskLowWorkerCap {
			return diskLowWorkerCap
		}
	}
	return ceiling
}

// SetFreeBytesFuncForTest injects a fake free-space probe and resets the last sample;
// the returned func restores the real probe (tests only).
func SetFreeBytesFuncForTest(fn func(dir string) (uint64, error)) func() {
	diskMu.Lock()
	prev := freeBytesFunc
	freeBytesFunc = fn
	diskLast = DiskSpaceSnapshot{Level: DiskOK}
	diskMu.Unlock()
	return func() {
		diskMu.Lock()
		freeBytesFunc = prev
		diskLast = DiskSpaceSnapshot{Level: DiskOK}
		diskMu.Unlock()
	}
}
