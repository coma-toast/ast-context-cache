package embedqueue

import (
	"log"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const missingVectorsSQL = `
SELECT DISTINCT s.file, s.project_path
FROM symbols s
LEFT JOIN vectors v ON v.symbol_id = s.id AND v.project_path = s.project_path AND COALESCE(v.doc_type, 'code') = 'code'
WHERE v.id IS NULL`

var (
	syncPendingMu    sync.Mutex
	syncPendingTimer *time.Timer
)

// SyncPendingFromDB marks indexed files that lack code vectors as pending retry.
func SyncPendingFromDB() int {
	rows, err := db.DB.Query(missingVectorsSQL)
	if err != nil {
		log.Printf("embedqueue: sync pending: %v", err)
		return 0
	}
	defer rows.Close()
	added := 0
	for rows.Next() {
		var file, projectPath string
		if err := rows.Scan(&file, &projectPath); err != nil {
			continue
		}
		if indexer.ShouldSkipEmbed(file) {
			continue
		}
		if markPendingIfNew(job{file: file, projectPath: projectPath}) {
			added++
		}
	}
	if added > 0 {
		realtime.Notify(realtime.EmbedFinished)
	}
	return added
}

func markPendingIfNew(j job) bool {
	pendingMu.Lock()
	defer pendingMu.Unlock()
	if pending == nil {
		pending = map[string]job{}
	}
	k := jobKey(j)
	if _, ok := pending[k]; ok {
		return false
	}
	pending[k] = j
	return true
}

// ScheduleSyncPending debounces a DB scan so pending reflects files missing vectors during outages.
func ScheduleSyncPending() {
	syncPendingMu.Lock()
	defer syncPendingMu.Unlock()
	if syncPendingTimer != nil {
		syncPendingTimer.Stop()
	}
	syncPendingTimer = time.AfterFunc(2*time.Second, func() {
		if n := SyncPendingFromDB(); n > 0 {
			log.Printf("embedqueue: %d files missing vectors marked pending", n)
		}
	})
}

// RecoverAfterEmbedder rescans for missing vectors and re-queues all pending embed jobs.
func RecoverAfterEmbedder() {
	added := SyncPendingFromDB()
	total := PendingCount()
	if added > 0 || total > 0 {
		log.Printf("embed recovery: %d newly synced, %d pending — re-queueing", added, total)
	}
	FlushPending()
}
