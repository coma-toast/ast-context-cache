package db

import (
	"log"
	"strings"
	"sync/atomic"
	"time"
)

const (
	walModerateBytes = 64 * 1024 * 1024
	walWarnBytes     = 128 * 1024 * 1024
	walHighBytes     = 256 * 1024 * 1024
)

var dbLockStreak atomic.Int32

// WalPressure returns ok, warn, or high based on index.db WAL size (primary growth source).
func WalPressure() string {
	wal := IndexWalBytes()
	switch {
	case wal >= walHighBytes:
		return "high"
	case wal >= walWarnBytes:
		return "warn"
	default:
		return "ok"
	}
}

// ShouldThrottleHeavyWork is true when index.db WAL is large enough to risk lock contention.
func ShouldThrottleHeavyWork() bool {
	return IndexWalBytes() >= walWarnBytes
}

// ThrottledEmbedWorkers caps worker count under WAL pressure.
func ThrottledEmbedWorkers(requested int) int {
	if requested < 1 {
		return requested
	}
	wal := IndexWalBytes()
	switch {
	case wal >= walHighBytes:
		if requested > 2 {
			return 2
		}
	case wal >= walWarnBytes:
		if requested > 4 {
			return 4
		}
	case wal >= walModerateBytes:
		if requested > 8 {
			return 8
		}
	}
	return requested
}

// NoteDBLock increments lock-contention tracking; call from retry paths on "database is locked".
func NoteDBLock() {
	n := dbLockStreak.Add(1)
	if n == 5 || n%20 == 0 {
		log.Printf("db pressure: database locked streak=%d index_wal=%s", n, FormatFileSize(IndexWalBytes()))
	}
}

// NoteDBOK clears lock-contention tracking after a successful write.
func NoteDBOK() {
	dbLockStreak.Store(0)
}

// IsDBLocked reports whether err is SQLite busy/locked.
func IsDBLocked(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "database is locked") || strings.Contains(msg, "sqlite_busy")
}

// StartPressureRelief is deprecated; WAL maintenance runs from StartWALCheckpoint.
func StartPressureRelief() {}

func retryQueryRetention(label string) {
	for attempt := 0; attempt < 6; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt*5) * time.Second)
		}
		n := RunQueryRetention()
		if n >= 0 {
			if n > 0 {
				log.Printf("query retention: %s deleted %d rows", label, n)
			}
			return
		}
	}
	log.Printf("query retention: %s deferred (database busy)", label)
}
