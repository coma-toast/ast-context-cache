package embedqueue

import (
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	selectMissingVectorsQuery = `
SELECT DISTINCT file, project_path FROM (
	SELECT DISTINCT s.file, s.project_path
	FROM symbols s
	LEFT JOIN vectors v
		ON v.symbol_id = s.id
		AND v.project_path = s.project_path
		AND COALESCE(v.doc_type, 'code') = 'code'
		AND v.content_hash = s.embed_hash
	WHERE v.id IS NULL
		AND s.embed_hash IS NOT NULL AND s.embed_hash != ''
	UNION
	SELECT DISTINCT s.file, s.project_path
	FROM symbols s
	LEFT JOIN vectors v
		ON v.symbol_id = s.id
		AND v.project_path = s.project_path
		AND COALESCE(v.doc_type, 'code') = 'code'
	WHERE v.id IS NULL
		AND (s.embed_hash IS NULL OR s.embed_hash = '')
)`
	deleteStaleEmbedPendingQuery = `
		DELETE FROM embed_pending
		WHERE NOT EXISTS (
			SELECT 1 FROM symbols s
			WHERE s.file = embed_pending.file AND s.project_path = embed_pending.project_path
		)`
)

var (
	recoveryMu       sync.Mutex
	errorScanOnce    sync.Once
	pendingReconOnce sync.Once

	flushLogMu        sync.Mutex
	lastFlushLogAt    time.Time
	lastFlushPending  int
	lastFlushInFlight int64
)

// shouldLogFlush rate-limits identical flush logs while work is in flight.
// Skip when pending/inFlight are unchanged, inFlight > 0, and last log was <10s ago.
func shouldLogFlush(pending int, inFlight int64) bool {
	flushLogMu.Lock()
	defer flushLogMu.Unlock()
	now := time.Now()
	same := pending == lastFlushPending && inFlight == lastFlushInFlight
	if same && inFlight > 0 && now.Sub(lastFlushLogAt) < 10*time.Second {
		return false
	}
	lastFlushLogAt = now
	lastFlushPending = pending
	lastFlushInFlight = inFlight
	return true
}

// SyncPendingFromDB marks indexed files that lack or have stale code vectors as pending retry.
func SyncPendingFromDB() int {
	type row struct{ file, projectPath string }
	conn, err := db.IndexReader()
	if err != nil {
		logger.Warn("Failed to sync pending", "error", err)
		return 0
	}
	rows, err := conn.Query(selectMissingVectorsQuery)
	if err != nil {
		logger.Warn("Failed to sync pending", "error", err)
		return 0
	}
	var pendingRows []row
	for rows.Next() {
		var r row
		if err := rows.Scan(&r.file, &r.projectPath); err != nil {
			continue
		}
		pendingRows = append(pendingRows, r)
	}
	rows.Close()
	added := 0
	for _, r := range pendingRows {
		if indexer.ShouldSkipEmbed(r.file) {
			continue
		}
		if markPendingIfNew(job{file: r.file, projectPath: r.projectPath}, pendingReasonSync) {
			added++
		}
	}
	if added > 0 {
		realtime.Notify(realtime.EmbedFinished)
	}
	flushPendingIfReady()
	return added
}

// StartErrorScanLoop periodically syncs pending from DB while the embedder is in error state.
func StartErrorScanLoop() {
	errorScanOnce.Do(func() {
		go func() {
			ticker := time.NewTicker(30 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				state, _ := embedder.HealthState()
				if state != "error" {
					continue
				}
				if n := SyncPendingFromDB(); n > 0 {
					logger.Info("Error scan marked files pending", "files", n)
				}
				// Primary is down; still re-queue for aux (onnx) catch-up when available.
				flushPendingIfReady()
			}
		}()
	})
}

// FlushPendingIfReady re-queues pending files when the embedder is healthy.
func FlushPendingIfReady() {
	flushPendingIfReady()
}

// pendingRetryBlocked explains why pending files can't be re-queued right now, or "".
func pendingRetryBlocked() string {
	if MaintenancePaused() {
		return "embedding is paused for WAL maintenance"
	}
	if state, _ := embedder.HealthState(); state == "error" && !auxCanCatchUp() {
		return "the embedder is down; retry the embedder first"
	}
	return ""
}

// RetryPendingNow re-queues every file awaiting an embed retry immediately instead of
// waiting for the reconciler's next idle tick. It returns how many were queued, or why
// nothing could be.
func RetryPendingNow() (queued int, blocked string) {
	if blocked = pendingRetryBlocked(); blocked != "" {
		return 0, blocked
	}
	queued = PendingCount()
	if queued > 0 {
		logger.Info("Manual retry of pending", "pending", queued)
		FlushPending()
	}
	return queued, ""
}

func flushPendingIfReady() {
	if pendingRetryBlocked() != "" {
		return
	}
	state, _ := embedder.HealthState()
	if PendingCount() == 0 {
		return
	}
	s := Snapshot()
	if shouldLogFlush(s.Pending, s.InFlight) {
		if state == "error" {
			logger.Info("Flush pending via aux catch-up", "pending", s.Pending, "queued", s.Queued, "in_flight", s.InFlight)
		} else {
			logger.Info("Flush pending", "pending", s.Pending, "queued", s.Queued, "in_flight", s.InFlight)
		}
	}
	FlushPending()
}

// StartPendingReconciler periodically re-flushes pending when the queue is idle but backlog remains.
func StartPendingReconciler() {
	pendingReconOnce.Do(func() {
		go func() {
			ticker := time.NewTicker(15 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				state, _ := embedder.HealthState()
				if state == "error" && !auxCanCatchUp() {
					continue
				}
				s := Snapshot()
				if s.Pending > 0 && s.Queued == 0 && s.InFlight == 0 {
					flushPendingIfReady()
				}
			}
		}()
	})
}

func recoverPending() {
	recoveryMu.Lock()
	defer recoveryMu.Unlock()
	purged := search.PurgeOrphanCodeVectors()
	pruned := pruneStaleEmbedPending()
	added := syncPendingFromDBLocked()
	total := PendingCount()
	s := Snapshot()
	if added > 0 || total > 0 || purged > 0 || pruned > 0 {
		logger.Info("Embed recovery", "synced", added, "pending", total, "purged_orphans", purged, "pruned_pending", pruned,
			"queued", s.Queued, "in_flight", s.InFlight)
	}
	flushPendingIfReady()
}

func syncPendingFromDBLocked() int {
	type row struct{ file, projectPath string }
	conn, err := db.IndexReader()
	if err != nil {
		logger.Warn("Failed to sync pending", "error", err)
		return 0
	}
	rows, err := conn.Query(selectMissingVectorsQuery)
	if err != nil {
		logger.Warn("Failed to sync pending", "error", err)
		return 0
	}
	var pendingRows []row
	for rows.Next() {
		var r row
		if err := rows.Scan(&r.file, &r.projectPath); err != nil {
			continue
		}
		pendingRows = append(pendingRows, r)
	}
	rows.Close()
	added := 0
	for _, r := range pendingRows {
		if indexer.ShouldSkipEmbed(r.file) {
			continue
		}
		if markPendingIfNew(job{file: r.file, projectPath: r.projectPath}, pendingReasonSync) {
			added++
		}
	}
	if added > 0 {
		realtime.Notify(realtime.EmbedFinished)
	}
	return added
}

func pruneStaleEmbedPending() int {
	conn, err := db.IndexReader()
	if err != nil {
		logger.Warn("Failed to prune embed_pending", "error", err)
		return 0
	}
	res, err := conn.Exec(deleteStaleEmbedPendingQuery)
	if err != nil {
		logger.Warn("Failed to prune embed_pending", "error", err)
		return 0
	}
	n, _ := res.RowsAffected()
	return int(n)
}

// RecoverAfterEmbedder rescans for missing vectors and re-queues all pending embed jobs.
func RecoverAfterEmbedder() {
	recoverPending()
}
