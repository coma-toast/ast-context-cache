package embedqueue

import (
	"log"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

const (
	pendingReasonFailed  = "failed"
	pendingReasonSync    = "sync"
	pendingReasonDrained = "drained"
)

func markPendingIfNew(j job, reason string) bool {
	pendingMu.Lock()
	if pending == nil {
		pending = map[string]job{}
	}
	k := jobKey(j)
	if _, ok := pending[k]; ok {
		pendingMu.Unlock()
		return false
	}
	pending[k] = j
	persistPending(j, reason)
	pendingMu.Unlock()
	return true
}

func persistPending(j job, reason string) {
	if db.DB == nil {
		return
	}
	_, err := db.DB.Exec(`INSERT OR REPLACE INTO embed_pending (file, project_path, reason, updated_at) VALUES (?, ?, ?, ?)`,
		j.file, j.projectPath, reason, time.Now().Unix())
	if err != nil {
		log.Printf("embedqueue: persist pending %s: %v", j.file, err)
	}
}

func clearPendingDB(j job) {
	if db.DB == nil {
		return
	}
	db.DB.Exec(`DELETE FROM embed_pending WHERE file = ? AND project_path = ?`, j.file, j.projectPath)
}

// LoadPendingFromDB hydrates the in-memory pending map from SQLite.
func LoadPendingFromDB() int {
	if db.DB == nil {
		return 0
	}
	rows, err := db.DB.Query(`SELECT file, project_path FROM embed_pending`)
	if err != nil {
		log.Printf("embedqueue: load pending: %v", err)
		return 0
	}
	defer rows.Close()
	loaded := 0
	pendingMu.Lock()
	if pending == nil {
		pending = map[string]job{}
	}
	for rows.Next() {
		var file, projectPath string
		if err := rows.Scan(&file, &projectPath); err != nil {
			continue
		}
		if indexer.ShouldSkipEmbed(file) {
			continue
		}
		k := jobKey(job{file: file, projectPath: projectPath})
		if _, ok := pending[k]; ok {
			continue
		}
		pending[k] = job{file: file, projectPath: projectPath}
		loaded++
	}
	pendingMu.Unlock()
	if loaded > 0 {
		log.Printf("embedqueue: loaded %d pending from DB", loaded)
	}
	return loaded
}
