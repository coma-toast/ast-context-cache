package db

import (
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	walModerateBytes = 64 * 1024 * 1024
	walWarnBytes     = 128 * 1024 * 1024
	walHighBytes     = 256 * 1024 * 1024

	// walDrainProgressBytes is the shrink between samples that counts as the WAL draining.
	walDrainProgressBytes = 4 * 1024 * 1024
)

var dbLockStreak atomic.Int32

var (
	backpressureMu  sync.Mutex
	backpressureCap = -1
	lastWalSample   int64
)

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

// UpdateWalBackpressure samples index.db WAL and returns the embed worker ceiling for all
// pools (-1 = no ceiling). The ceiling keeps stepping down — to 0 if needed — while the WAL
// holds or grows, and climbs back toward the static throttle once the WAL starts draining.
func UpdateWalBackpressure(poolTarget int) int {
	wal := IndexWalBytes()
	static := ThrottledEmbedWorkers(poolTarget)
	backpressureMu.Lock()
	defer backpressureMu.Unlock()
	lastWal := lastWalSample
	lastWalSample = wal
	if wal < walModerateBytes {
		backpressureCap = -1
		return -1
	}
	backpressureCap = nextBackpressureCap(backpressureCap, static, wal, lastWal)
	return backpressureCap
}

// nextBackpressureCap advances the worker ceiling for one WAL sample.
func nextBackpressureCap(prevCap, static int, wal, lastWal int64) int {
	if static < 0 {
		static = 0
	}
	switch {
	case prevCap < 0:
		return static
	case lastWal <= 0:
		return min(prevCap, static)
	case wal <= lastWal-walDrainProgressBytes:
		return stepCapUp(prevCap, static)
	default:
		return stepCapDown(min(prevCap, static))
	}
}

func stepCapDown(n int) int {
	if n <= 1 {
		return 0
	}
	return n / 2
}

func stepCapUp(n, static int) int {
	if n < 1 {
		return min(1, static)
	}
	return min(n*2, static)
}

// WalBackpressureCap returns the current WAL worker ceiling without resampling (-1 = none).
func WalBackpressureCap() int {
	backpressureMu.Lock()
	defer backpressureMu.Unlock()
	return backpressureCap
}

// WalBackpressurePaused reports whether WAL backpressure is holding embed workers at 0.
func WalBackpressurePaused() bool {
	return WalBackpressureCap() == 0
}

// CappedEmbedWorkers applies the static WAL throttle plus any active backpressure ceiling.
func CappedEmbedWorkers(requested int) int {
	n := ThrottledEmbedWorkers(requested)
	if ceiling := WalBackpressureCap(); ceiling >= 0 && ceiling < n {
		return ceiling
	}
	return n
}

// SetWalBackpressureForTest overrides adaptive throttle state (tests only).
func SetWalBackpressureForTest(ceiling int, lastWal int64) {
	backpressureMu.Lock()
	defer backpressureMu.Unlock()
	backpressureCap = ceiling
	lastWalSample = lastWal
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
