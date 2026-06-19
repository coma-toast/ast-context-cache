package db

import (
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	walWarnBytes = 256 * 1024 * 1024
	walHighBytes = 512 * 1024 * 1024
)

var (
	dbLockStreak atomic.Int32
	pressureMu   sync.Mutex
	lastRelief   time.Time
)

// WalPressure returns ok, warn, or high based on WAL file size.
func WalPressure() string {
	wal := WalFileBytes()
	switch {
	case wal >= walHighBytes:
		return "high"
	case wal >= walWarnBytes:
		return "warn"
	default:
		return "ok"
	}
}

// ShouldThrottleHeavyWork is true when SQLite WAL is large enough to risk lock contention.
func ShouldThrottleHeavyWork() bool {
	return WalFileBytes() >= walWarnBytes
}

// ThrottledEmbedWorkers caps worker count under WAL pressure.
func ThrottledEmbedWorkers(requested int) int {
	if requested < 1 {
		return requested
	}
	wal := WalFileBytes()
	switch {
	case wal >= walHighBytes:
		if requested > 2 {
			return 2
		}
	case wal >= walWarnBytes:
		if requested > 4 {
			return 4
		}
	}
	return requested
}

// NoteDBLock increments lock-contention tracking; call from retry paths on "database is locked".
func NoteDBLock() {
	n := dbLockStreak.Add(1)
	if n == 5 || n%20 == 0 {
		log.Printf("db pressure: database locked streak=%d wal=%s", n, FormatFileSize(WalFileBytes()))
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

// StartPressureRelief runs periodic relief when the WAL is large.
func StartPressureRelief() {
	go func() {
		ticker := time.NewTicker(2 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			relievePressure()
		}
	}()
}

func relievePressure() {
	wal := WalFileBytes()
	if wal < walWarnBytes {
		return
	}
	pressureMu.Lock()
	if time.Since(lastRelief) < time.Minute {
		pressureMu.Unlock()
		return
	}
	lastRelief = time.Now()
	pressureMu.Unlock()
	if busy, frames, ckpt, err := CheckpointWAL(true); err == nil && frames > 0 {
		log.Printf("pressure relief: WAL checkpoint busy=%d log=%d checkpointed=%d wal=%s", busy, frames, ckpt, FormatFileSize(WalFileBytes()))
	}
	if wal >= walHighBytes {
		RunQueryRetention()
	}
}

func retryQueryRetention(label string) {
	for attempt := 0; attempt < 6; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt*5) * time.Second)
		}
		before := dbLockStreak.Load()
		n := RunQueryRetention()
		if n > 0 || dbLockStreak.Load() == before {
			if n > 0 {
				log.Printf("query retention: %s deleted %d rows", label, n)
			}
			return
		}
	}
	log.Printf("query retention: %s deferred (database busy)", label)
}
