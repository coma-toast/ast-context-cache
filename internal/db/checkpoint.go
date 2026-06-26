package db

import (
	"database/sql"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

const (
	walPassiveBytes  = 32 * 1024 * 1024
	walTruncateBytes = 64 * 1024 * 1024
	walForceBytes    = 128 * 1024 * 1024

	checkpointOpTimeout  = 20 * time.Second
	checkpointMaxElapsed = 90 * time.Second
)

var (
	checkpointPools []*sql.DB
	walBusyStreak   atomic.Int32

	checkpointMu  sync.Mutex
	lastMaintAt   time.Time
	lastMaintBusy bool

	// BeforeForceCheckpoint quiesces heavy DB users (embed workers, vector cache).
	// Set from cmd/ast-mcp/main.go to avoid import cycles.
	BeforeForceCheckpoint func()
	AfterForceCheckpoint  func()
)

func openCheckpointPools() error {
	paths := []string{indexDBPath(), usageDBPath(), contextDBPath()}
	checkpointPools = make([]*sql.DB, 0, len(paths))
	for _, p := range paths {
		conn, err := sql.Open("sqlite3", p+"?_journal_mode=WAL&_busy_timeout=15000")
		if err != nil {
			closeCheckpointPools()
			return err
		}
		conn.SetMaxOpenConns(1)
		conn.SetMaxIdleConns(1)
		conn.Exec(`PRAGMA busy_timeout=15000`)
		checkpointPools = append(checkpointPools, conn)
	}
	return nil
}

func closeCheckpointPools() {
	for _, c := range checkpointPools {
		if c != nil {
			c.Close()
		}
	}
	checkpointPools = nil
}

func checkpointOn(dbConn *sql.DB, mode string) (busy, walFrames, checkpointed int, err error) {
	if checkpointAbort.Load() {
		return 1, 0, 0, fmt.Errorf("checkpoint aborted")
	}
	if WALMaintenanceActive() {
		setWALPhase(WALPhaseCheckpoint, mode)
	}
	type res struct {
		busy, frames, ckpt int
		err                error
	}
	ch := make(chan res, 1)
	go func() {
		row := dbConn.QueryRow("PRAGMA wal_checkpoint(" + mode + ")")
		var b, f, c int
		e := row.Scan(&b, &f, &c)
		ch <- res{b, f, c, e}
	}()
	select {
	case r := <-ch:
		busy, walFrames, checkpointed, err = r.busy, r.frames, r.ckpt, r.err
	case <-time.After(checkpointOpTimeout):
		busy, err = 1, fmt.Errorf("checkpoint %s timed out after %s", mode, checkpointOpTimeout)
	}
	if WALMaintenanceActive() {
		recordWALCheckpointResult(busy, walFrames, checkpointed, err)
	}
	return
}

func checkpointAll(mode string) (busy, walFrames, checkpointed int, err error) {
	for i, conn := range checkpointPools {
		b, f, c, e := checkpointOn(conn, mode)
		if e != nil && err == nil {
			err = e
		}
		if b > busy {
			busy = b
		}
		walFrames += f
		checkpointed += c
		if b == 1 && i == 0 {
			break
		}
	}
	return
}

func prepCheckpoint(pauseWriters bool) (restore func()) {
	restore = func() {}
	if !pauseWriters {
		return
	}
	setWALPhase(WALPhasePausing, "")
	if BeforeForceCheckpoint != nil {
		BeforeForceCheckpoint()
	}
	setWALPhase(WALPhaseDraining, "")
	FlushWriteBuffers()
	FlushIndexWriter()
	time.Sleep(300 * time.Millisecond)
	if AfterForceCheckpoint != nil {
		restore = AfterForceCheckpoint
	}
	return
}

func logCheckpoint(label string, walBefore int64, busy, walFrames, checkpointed int) {
	if busy == 0 && walFrames == 0 && checkpointed == 0 {
		return
	}
	log.Printf("WAL %s: busy=%d log=%d checkpointed=%d wal %s -> %s",
		label, busy, walFrames, checkpointed, FormatFileSize(walBefore), FormatFileSize(walFileBytes()))
}

// CheckpointWAL flushes the WAL on dedicated connections. truncate=true pauses writers when wal is large.
func CheckpointWAL(truncate bool) (busy, walFrames, checkpointed int, err error) {
	if truncate {
		return maintainWAL("truncate", false)
	}
	return checkpointAll("PASSIVE")
}

func maintainWAL(reason string, force bool) (busy, walFrames, checkpointed int, err error) {
	checkpointMu.Lock()
	defer checkpointMu.Unlock()

	if checkpointAbort.Load() {
		return 1, 0, 0, fmt.Errorf("checkpoint aborted")
	}

	wal := walFileBytes()
	if !force && time.Since(lastMaintAt) < 45*time.Second && !lastMaintBusy {
		return 0, 0, 0, nil
	}

	beginWALMaintenance(reason)
	defer endWALMaintenance()

	deadline := time.Now().Add(checkpointMaxElapsed)
	walBefore := wal
	pause := force || wal >= walTruncateBytes
	restore := prepCheckpoint(pause)
	defer func() {
		setWALPhase(WALPhaseRestoring, "")
		restore()
	}()

	tryCheckpoint := func(mode string) (int, int, int, error) {
		if checkpointAbort.Load() || time.Now().After(deadline) {
			return 1, 0, 0, fmt.Errorf("checkpoint aborted or deadline exceeded")
		}
		return checkpointAll(mode)
	}

	useForcePath := force || wal >= walForceBytes || walBusyStreak.Load() >= 3
	if useForcePath {
		busy, walFrames, checkpointed, err = tryCheckpoint("RESTART")
		if err != nil || busy == 0 {
			walBusyStreak.Store(0)
			logCheckpoint("force RESTART ("+reason+")", walBefore, busy, walFrames, checkpointed)
			lastMaintAt = time.Now()
			lastMaintBusy = busy == 1
			return
		}
		busy, walFrames, checkpointed, err = tryCheckpoint("TRUNCATE")
		if busy == 0 {
			walBusyStreak.Store(0)
		}
		logCheckpoint("force TRUNCATE ("+reason+")", walBefore, busy, walFrames, checkpointed)
		lastMaintAt = time.Now()
		lastMaintBusy = busy == 1
		return
	}

	busy, walFrames, checkpointed, err = tryCheckpoint("TRUNCATE")
	trackCheckpointResult(true, busy, walFrames)
	if busy == 1 && time.Now().Before(deadline) && !checkpointAbort.Load() {
		busy, walFrames, checkpointed, err = tryCheckpoint("RESTART")
		if busy == 1 && time.Now().Before(deadline) && !checkpointAbort.Load() {
			busy, walFrames, checkpointed, err = tryCheckpoint("TRUNCATE")
		}
		trackCheckpointResult(true, busy, walFrames)
	}
	if busy == 0 {
		walBusyStreak.Store(0)
	}
	logCheckpoint("TRUNCATE ("+reason+")", walBefore, busy, walFrames, checkpointed)
	lastMaintAt = time.Now()
	lastMaintBusy = busy == 1
	return
}

// ForceCheckpointWAL pauses heavy writers/readers via hooks, flushes buffers, then
// runs RESTART (fall back to TRUNCATE) on dedicated connections.
func ForceCheckpointWAL() (busy, walFrames, checkpointed int, err error) {
	return maintainWAL("force", true)
}

func shouldForceCheckpoint(wal int64) bool {
	streak := walBusyStreak.Load()
	return wal >= walForceBytes || streak >= 3
}

func maybeForceCheckpoint(wal int64, reason string) bool {
	if !shouldForceCheckpoint(wal) {
		return false
	}
	log.Printf("WAL force checkpoint (%s): wal=%s busy_streak=%d", reason, FormatFileSize(wal), walBusyStreak.Load())
	maintainWAL("busy", true)
	return true
}

func trackCheckpointResult(truncate bool, busy, walFrames int) {
	if !truncate {
		return
	}
	if busy == 1 {
		walBusyStreak.Add(1)
		return
	}
	walBusyStreak.Store(0)
}

func runPassiveCheckpoint() {
	checkpointMu.Lock()
	defer checkpointMu.Unlock()
	if time.Since(lastMaintAt) < 15*time.Second {
		return
	}
	walBefore := walFileBytes()
	busy, frames, ckpt, err := checkpointAll("PASSIVE")
	if err != nil {
		return
	}
	if frames > 0 || ckpt > 0 {
		logCheckpoint("PASSIVE", walBefore, busy, frames, ckpt)
	}
	lastMaintAt = time.Now()
	lastMaintBusy = busy == 1
}

func runWALMaintenanceCycle(reason string) {
	wal := walFileBytes()
	if wal < walPassiveBytes {
		return
	}
	if wal > walTruncateBytes {
		busy, frames, _, err := maintainWAL(reason, false)
		if err == nil && busy == 1 {
			maybeForceCheckpoint(wal, "busy")
		} else if err == nil {
			_ = frames
		}
		if wal >= walHighBytes {
			retryQueryRetention("pressure")
		}
		return
	}
	runPassiveCheckpoint()
}
