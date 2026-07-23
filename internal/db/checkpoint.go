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
	embedDeferDuration     = 2 * time.Minute
	quietWALCooldown        = 5 * time.Minute
)

var (
	walBusyStreak atomic.Int32

	checkpointMu  sync.Mutex
	lastMaintAt   time.Time
	lastMaintBusy bool

	maintBusyCycles    int
	maintBackoffUntil  time.Time
	truncateDeferUntil time.Time
	truncateDeferLogged bool

	lastSkipLogAt time.Time

	BeforeForceCheckpoint func()
	AfterForceCheckpoint  func()

	quietWALMu     sync.Mutex
	lastQuietWALAt time.Time
)

func logPoolStats(name string, pool *sql.DB) {
	if pool == nil {
		log.Printf("WAL diagnostics: pool=%s unavailable", name)
		return
	}
	s := pool.Stats()
	log.Printf(
		"WAL diagnostics: pool=%s open=%d in_use=%d idle=%d wait_count=%d wait_dur=%s max_idle_closed=%d max_idle_time_closed=%d max_lifetime_closed=%d",
		name, s.OpenConnections, s.InUse, s.Idle, s.WaitCount, s.WaitDuration, s.MaxIdleClosed, s.MaxIdleTimeClosed, s.MaxLifetimeClosed,
	)
}

func logBusyCheckpointDiagnostics(path, mode string, walFrames, checkpointed int) {
	label := dbLabel(path)
	inFlight := int64(-1)
	if WALInFlightHook != nil {
		inFlight = WALInFlightHook()
	}
	embedIdle := true
	if EmbedQueueIdleHook != nil {
		embedIdle = EmbedQueueIdleHook()
	}
	log.Printf(
		"WAL diagnostics: busy checkpoint db=%s mode=%s log=%d checkpointed=%d index_wal=%s usage_wal=%s context_wal=%s index_read_quiesced=%t embed_idle=%t in_flight=%d",
		label, mode, walFrames, checkpointed, FormatFileSize(IndexWalBytes()), FormatFileSize(UsageWalBytes()), FormatFileSize(ContextWalBytes()), IndexReadQuiesced(), embedIdle, inFlight,
	)
	logPoolStats("index", IndexDB)
	logPoolStats("usage", DB)
	logPoolStats("context", ContextDB)
}

func shouldDeferMaint() bool {
	return time.Now().Before(maintBackoffUntil)
}

func noteMaintResult(busy int) {
	setWalSkipReason("", time.Time{})
	if busy == 1 {
		maintBusyCycles++
		switch {
		case maintBusyCycles >= 3:
			maintBackoffUntil = time.Now().Add(15 * time.Minute)
		default:
			maintBackoffUntil = time.Now().Add(5 * time.Minute)
		}
		setWalSkipReason("busy_readers", maintBackoffUntil)
		log.Printf("WAL TRUNCATE deferred: readers busy (retry after %s)", maintBackoffUntil.Format(time.RFC3339))
		return
	}
	maintBusyCycles = 0
	maintBackoffUntil = time.Time{}
}

func logWalSkip(reason string, detail string) {
	now := time.Now()
	if reason == "embed_deferred" && now.Sub(lastSkipLogAt) < time.Minute {
		return
	}
	lastSkipLogAt = now
	if detail != "" {
		log.Printf("WAL maintenance skipped (%s): %s", reason, detail)
	} else {
		log.Printf("WAL maintenance skipped (%s)", reason)
	}
}

func checkpointFile(path, mode string) (busy, walFrames, checkpointed int, err error) {
	if checkpointAbort.Load() {
		return 1, 0, 0, fmt.Errorf("checkpoint aborted")
	}
	if WALMaintenanceActive() {
		setWALPhase(WALPhaseCheckpoint, mode)
	}
	conn, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=15000")
	if err != nil {
		return 0, 0, 0, err
	}
	conn.SetMaxOpenConns(1)
	type res struct {
		busy, frames, ckpt int
		err                error
	}
	ch := make(chan res, 1)
	go func() {
		row := conn.QueryRow("PRAGMA wal_checkpoint(" + mode + ")")
		var b, f, c int
		e := row.Scan(&b, &f, &c)
		ch <- res{b, f, c, e}
	}()
	select {
	case r := <-ch:
		conn.Close()
		busy, walFrames, checkpointed, err = r.busy, r.frames, r.ckpt, r.err
	case <-time.After(checkpointOpTimeout):
		conn.Close()
		busy, err = 1, fmt.Errorf("checkpoint %s timed out after %s", mode, checkpointOpTimeout)
	}
	if WALMaintenanceActive() {
		recordWALCheckpointResult(busy, walFrames, checkpointed, err)
	}
	if busy == 1 {
		logBusyCheckpointDiagnostics(path, mode, walFrames, checkpointed)
	}
	return
}

func checkpointIndexTruncate() (busy, walFrames, checkpointed int, err error) {
	if err := quiesceIndexPool(); err != nil {
		return 1, 0, 0, err
	}
	defer func() {
		if rerr := restoreIndexPool(); rerr != nil {
			log.Printf("WAL: restore index pool: %v", rerr)
		}
	}()
	return checkpointFile(indexDBPath(), "TRUNCATE")
}

func checkpointAllDbs(mode string, indexQuiesce bool) (busy, walFrames, checkpointed int, err error) {
	if mode == "TRUNCATE" && indexQuiesce {
		b, f, c, e := checkpointIndexTruncate()
		if e != nil && err == nil {
			err = e
		}
		if b > busy {
			busy = b
		}
		walFrames += f
		checkpointed += c
		if busy == 1 {
			return
		}
	} else {
		b, f, c, e := checkpointFile(indexDBPath(), mode)
		if e != nil && err == nil {
			err = e
		}
		if b > busy {
			busy = b
		}
		walFrames += f
		checkpointed += c
		if busy == 1 {
			return
		}
	}
	for _, path := range []string{usageDBPath(), contextDBPath()} {
		b, f, c, e := checkpointFile(path, mode)
		if e != nil && err == nil {
			err = e
		}
		if b > busy {
			busy = b
		}
		walFrames += f
		checkpointed += c
		if busy == 1 {
			break
		}
	}
	return
}

func prepCheckpoint(pauseWriters bool) (restore func(), ready bool) {
	restore = func() {}
	if !pauseWriters {
		return restore, true
	}
	setWALPhase(WALPhasePausing, "")
	if BeforeForceCheckpoint != nil {
		BeforeForceCheckpoint()
	}
	// Bind restore immediately so an abort after pause still unpauses embed workers
	// (maintainWAL defers restore(); previously an in-flight abort left workers stuck at 0).
	if AfterForceCheckpoint != nil {
		restore = AfterForceCheckpoint
	}
	if WALInFlightHook != nil {
		if n := WALInFlightHook(); n > 0 {
			log.Printf("WAL: checkpoint blocked — %d embed job(s) still in-flight after drain", n)
			setWalSkipReason("embed_in_flight", time.Now().Add(2*time.Minute))
			return restore, false
		}
	}
	setWALPhase(WALPhaseDraining, "")
	FlushWriteBuffers()
	FlushIndexWriter()
	time.Sleep(300 * time.Millisecond)
	return restore, true
}

func logCheckpoint(label string, walBefore int64, busy, walFrames, checkpointed int) {
	log.Printf("WAL %s: busy=%d log=%d checkpointed=%d index_wal %s -> %s total_wal %s",
		label, busy, walFrames, checkpointed,
		FormatFileSize(walBefore), FormatFileSize(IndexWalBytes()), FormatFileSize(walFileBytes()))
}

func CheckpointWAL(truncate bool) (busy, walFrames, checkpointed int, err error) {
	if truncate {
		return maintainWAL("truncate", false)
	}
	if IndexWalBytes() >= walPassiveBytes {
		return checkpointFile(indexDBPath(), "PASSIVE")
	}
	return 0, 0, 0, nil
}

func maintainWAL(reason string, force bool) (busy, walFrames, checkpointed int, err error) {
	checkpointMu.Lock()
	defer checkpointMu.Unlock()

	if checkpointAbort.Load() {
		logWalSkip("aborted", "shutdown requested")
		return 1, 0, 0, fmt.Errorf("checkpoint aborted")
	}
	if !force && shouldDeferMaint() {
		setWalSkipReason("backoff", maintBackoffUntil)
		logWalSkip("backoff", "until "+maintBackoffUntil.Format(time.RFC3339))
		return 0, 0, 0, nil
	}

	indexWal := IndexWalBytes()
	if !force && time.Since(lastMaintAt) < 45*time.Second && !lastMaintBusy {
		setWalSkipReason("throttle_45s", lastMaintAt.Add(45*time.Second))
		return 0, 0, 0, nil
	}

	if indexWal >= walTruncateBytes {
		log.Printf("WAL maintenance starting reason=%s force=%v index_wal=%s", reason, force, FormatFileSize(indexWal))
	}
	setWalSkipReason("", time.Time{})
	truncateDeferLogged = false

	beginWALMaintenance(reason)
	defer endWALMaintenance()

	deadline := time.Now().Add(checkpointMaxElapsed)
	walBefore := indexWal
	pause := force || indexWal >= walTruncateBytes
	if pause {
		indexReadGate.Store(true)
	}
	restore, ready := prepCheckpoint(pause)
	defer func() {
		setWALPhase(WALPhaseRestoring, "")
		restore()
		if pause && IndexDB != nil {
			indexReadGate.Store(false)
		}
	}()
	if pause && !ready {
		logCheckpoint("aborted (embed in-flight)", walBefore, 1, 0, 0)
		lastMaintAt = time.Now()
		lastMaintBusy = true
		noteMaintResult(1)
		return 1, 0, 0, fmt.Errorf("embed workers still active")
	}

	indexQuiesce := pause
	tryCheckpoint := func(mode string) (int, int, int, error) {
		if checkpointAbort.Load() || time.Now().After(deadline) {
			return 1, 0, 0, fmt.Errorf("checkpoint aborted or deadline exceeded")
		}
		return checkpointAllDbs(mode, indexQuiesce && mode == "TRUNCATE")
	}

	useForcePath := force || indexWal >= walForceBytes || walBusyStreak.Load() >= 3
	if useForcePath {
		rBusy, rFrames, rCkpt, rErr := tryCheckpoint("RESTART")
		walFrames += rFrames
		checkpointed += rCkpt
		busy = rBusy
		if rErr != nil {
			err = rErr
			logCheckpoint("force RESTART error ("+reason+")", walBefore, busy, walFrames, checkpointed)
			lastMaintAt = time.Now()
			lastMaintBusy = busy == 1
			noteMaintResult(busy)
			return
		}
		if indexWal >= walTruncateBytes || force {
			tBusy, tFrames, tCkpt, tErr := tryCheckpoint("TRUNCATE")
			walFrames += tFrames
			checkpointed += tCkpt
			if tErr != nil && err == nil {
				err = tErr
			}
			if tBusy > busy {
				busy = tBusy
			}
			if busy == 0 {
				walBusyStreak.Store(0)
			}
			logCheckpoint("force RESTART+TRUNCATE ("+reason+")", walBefore, busy, walFrames, checkpointed)
		} else {
			if busy == 0 {
				walBusyStreak.Store(0)
			}
			logCheckpoint("force RESTART ("+reason+")", walBefore, busy, walFrames, checkpointed)
		}
		lastMaintAt = time.Now()
		lastMaintBusy = busy == 1
		noteMaintResult(busy)
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
	noteMaintResult(busy)
	return
}

func ForceCheckpointWAL() (busy, walFrames, checkpointed int, err error) {
	return maintainWAL("force", true)
}

// TryQuietWALTruncate schedules a forced WAL TRUNCATE during a quiet period
// (embedder error, workers paused, queue idle). Debounced and size-gated.
func TryQuietWALTruncate(reason string) {
	indexWal := IndexWalBytes()
	quietWALMu.Lock()
	ok := shouldAttemptQuietWAL(time.Now(), indexWal, WALMaintenanceActive(), lastQuietWALAt)
	if ok {
		lastQuietWALAt = time.Now()
	}
	quietWALMu.Unlock()
	if !ok {
		return
	}
	go func() {
		log.Printf("WAL: quiet period (%s) — attempting TRUNCATE (index_wal=%s)", reason, FormatFileSize(indexWal))
		maintainWAL(reason, true)
	}()
}

// TryMaintainWALOnEmbedderError is kept as an alias for older call sites.
func TryMaintainWALOnEmbedderError() {
	TryQuietWALTruncate("embedder_error")
}

func shouldAttemptQuietWAL(now time.Time, indexWal int64, maintActive bool, lastAttempt time.Time) bool {
	if indexWal < walTruncateBytes {
		return false
	}
	if maintActive {
		return false
	}
	if !lastAttempt.IsZero() && now.Sub(lastAttempt) < quietWALCooldown {
		return false
	}
	return true
}

// Deprecated: use shouldAttemptQuietWAL.
func shouldAttemptWALOnEmbedderError(now time.Time, indexWal int64, maintActive bool, lastAttempt time.Time) bool {
	return shouldAttemptQuietWAL(now, indexWal, maintActive, lastAttempt)
}

func shouldForceCheckpoint(wal int64) bool {
	streak := walBusyStreak.Load()
	return wal >= walForceBytes || streak >= 3
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
	if IndexWalBytes() < walPassiveBytes {
		setWalSkipReason("below_threshold", time.Time{})
		return
	}
	if time.Since(lastMaintAt) < 15*time.Second {
		return
	}
	walBefore := IndexWalBytes()
	busy, frames, ckpt, err := checkpointFile(indexDBPath(), "PASSIVE")
	if err != nil {
		return
	}
	logCheckpoint("PASSIVE index", walBefore, busy, frames, ckpt)
	lastMaintAt = time.Now()
	lastMaintBusy = busy == 1
}

// truncateMaintForce returns true when maintenance should run with force=true after embed defer.
func truncateMaintForce(indexWal int64) (force bool, skip bool) {
	if indexWal <= walTruncateBytes {
		return false, true
	}
	if embedQueueIdle() {
		truncateDeferUntil = time.Time{}
		truncateDeferLogged = false
		return false, false
	}
	now := time.Now()
	if truncateDeferUntil.IsZero() {
		truncateDeferUntil = now.Add(embedDeferDuration)
		truncateDeferLogged = false
	}
	if now.Before(truncateDeferUntil) {
		setWalSkipReason("embed_deferred", truncateDeferUntil)
		if !truncateDeferLogged {
			truncateDeferLogged = true
			log.Printf("WAL: deferring TRUNCATE (embed active) until %s", truncateDeferUntil.Format(time.RFC3339))
		}
		return false, true
	}
	truncateDeferUntil = time.Time{}
	truncateDeferLogged = false
	log.Printf("WAL: embed still active after defer — forcing maintenance")
	return true, false
}

func runWALMaintenanceCycle(reason string) {
	if shouldDeferMaint() {
		setWalSkipReason("backoff", maintBackoffUntil)
		return
	}
	indexWal := IndexWalBytes()
	if indexWal < walPassiveBytes {
		setWalSkipReason("below_threshold", time.Time{})
		return
	}
	if indexWal > walTruncateBytes {
		force, skip := truncateMaintForce(indexWal)
		if skip {
			return
		}
		maintainWAL(reason, force)
		if indexWal >= walHighBytes {
			retryQueryRetention("pressure")
		}
		return
	}
	runPassiveCheckpoint()
}

func ResetMaintBackoffForTest() {
	maintBusyCycles = 0
	maintBackoffUntil = time.Time{}
	truncateDeferUntil = time.Time{}
	truncateDeferLogged = false
	quietWALMu.Lock()
	lastQuietWALAt = time.Time{}
	quietWALMu.Unlock()
}

// TruncateDeferUntilForTest exposes embed defer deadline (tests only).
func TruncateDeferUntilForTest() time.Time { return truncateDeferUntil }

// SetTruncateDeferUntilForTest sets embed defer deadline (tests only).
func SetTruncateDeferUntilForTest(t time.Time) { truncateDeferUntil = t }
