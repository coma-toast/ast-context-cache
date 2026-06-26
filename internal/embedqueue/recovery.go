package embedqueue

import (
	"log"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const missingVectorsSQL = `
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

var (
	recoveryMu        sync.Mutex
	errorScanOnce     sync.Once
	pendingReconOnce  sync.Once
)

// SyncPendingFromDB marks indexed files that lack or have stale code vectors as pending retry.
func SyncPendingFromDB() int {
	type row struct{ file, projectPath string }
	rows, err := db.IndexDB.Query(missingVectorsSQL)
	if err != nil {
		log.Printf("embedqueue: sync pending: %v", err)
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
					log.Printf("embedqueue: error-scan marked %d files pending", n)
				}
			}
		}()
	})
}

// FlushPendingIfReady re-queues pending files when the embedder is healthy.
func FlushPendingIfReady() {
	flushPendingIfReady()
}

func flushPendingIfReady() {
	state, _ := embedder.HealthState()
	if state == "error" {
		return
	}
	if PendingCount() == 0 {
		return
	}
	s := Snapshot()
	log.Printf("embedqueue: flush pending=%d queued=%d inFlight=%d", s.Pending, s.Queued, s.InFlight)
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
				if state == "error" {
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
		log.Printf("embed recovery: synced=%d pending=%d purged_orphans=%d pruned_pending=%d queued=%d inFlight=%d",
			added, total, purged, pruned, s.Queued, s.InFlight)
	}
	flushPendingIfReady()
}

func syncPendingFromDBLocked() int {
	type row struct{ file, projectPath string }
	rows, err := db.IndexDB.Query(missingVectorsSQL)
	if err != nil {
		log.Printf("embedqueue: sync pending: %v", err)
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
	res, err := db.IndexDB.Exec(`
		DELETE FROM embed_pending
		WHERE NOT EXISTS (
			SELECT 1 FROM symbols s
			WHERE s.file = embed_pending.file AND s.project_path = embed_pending.project_path
		)`)
	if err != nil {
		log.Printf("embedqueue: prune embed_pending: %v", err)
		return 0
	}
	n, _ := res.RowsAffected()
	return int(n)
}

// RecoverAfterEmbedder rescans for missing vectors and re-queues all pending embed jobs.
func RecoverAfterEmbedder() {
	recoverPending()
}
