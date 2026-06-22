package embedqueue

import (
	"log"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

const (
	pendingReasonFailed  = "failed"
	pendingReasonSync    = "sync"
	pendingReasonDrained = "drained"

	pendingFlushInterval = 3 * time.Second
	pendingFlushSize     = 40
)

type pendingDirtyRow struct {
	j      job
	reason string
}

var (
	pendingDirtyMu sync.Mutex
	pendingDirty   map[string]pendingDirtyRow
	pendingDelete  map[string]job
	pendingFlush   sync.Once
)

func markPendingIfNew(j job, reason string) bool {
	if isProjectCancelled(j.projectPath) {
		return false
	}
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
	enqueuePendingPersist(j, reason)
	pendingMu.Unlock()
	return true
}

func enqueuePendingPersist(j job, reason string) {
	pendingDirtyMu.Lock()
	if pendingDirty == nil {
		pendingDirty = map[string]pendingDirtyRow{}
	}
	delete(pendingDelete, jobKey(j))
	pendingDirty[jobKey(j)] = pendingDirtyRow{j: j, reason: reason}
	n := len(pendingDirty)
	pendingDirtyMu.Unlock()
	if n >= pendingFlushSize {
		go FlushPendingDB()
	}
}

func clearPendingDB(j job) {
	k := jobKey(j)
	pendingDirtyMu.Lock()
	if pendingDelete == nil {
		pendingDelete = map[string]job{}
	}
	delete(pendingDirty, k)
	pendingDelete[k] = j
	pendingDirtyMu.Unlock()
}

func startPendingDBFlusher() {
	pendingFlush.Do(func() {
		go func() {
			t := time.NewTicker(pendingFlushInterval)
			defer t.Stop()
			for range t.C {
				FlushPendingDB()
			}
		}()
	})
}

// FlushPendingDB commits batched embed_pending upserts and deletes.
func FlushPendingDB() {
	pendingDirtyMu.Lock()
	if len(pendingDirty) == 0 && len(pendingDelete) == 0 {
		pendingDirtyMu.Unlock()
		return
	}
	upserts := pendingDirty
	deletes := pendingDelete
	pendingDirty = map[string]pendingDirtyRow{}
	pendingDelete = map[string]job{}
	pendingDirtyMu.Unlock()

	tx, err := db.DB.Begin()
	if err != nil {
		requeuePendingFlush(upserts, deletes)
		return
	}
	defer tx.Rollback()

	upsertStmt, err := tx.Prepare(`INSERT OR REPLACE INTO embed_pending (file, project_path, reason, updated_at) VALUES (?, ?, ?, ?)`)
	if err != nil {
		requeuePendingFlush(upserts, deletes)
		return
	}
	defer upsertStmt.Close()

	now := time.Now().Unix()
	locked := false
	for _, row := range upserts {
		if _, err := upsertStmt.Exec(row.j.file, row.j.projectPath, row.reason, now); err != nil {
			if db.IsDBLocked(err) {
				db.NoteDBLock()
				locked = true
				pendingDirtyMu.Lock()
				if pendingDirty == nil {
					pendingDirty = map[string]pendingDirtyRow{}
				}
				k := jobKey(row.j)
				if _, ok := pendingDirty[k]; !ok {
					pendingDirty[k] = row
				}
				pendingDirtyMu.Unlock()
				continue
			}
			log.Printf("embedqueue: persist pending %s: %v", row.j.file, err)
		}
	}

	delStmt, err := tx.Prepare(`DELETE FROM embed_pending WHERE file = ? AND project_path = ?`)
	if err != nil {
		requeuePendingFlush(upserts, deletes)
		return
	}
	defer delStmt.Close()
	for _, j := range deletes {
		if _, err := delStmt.Exec(j.file, j.projectPath); err != nil {
			if db.IsDBLocked(err) {
				db.NoteDBLock()
				locked = true
				pendingDirtyMu.Lock()
				if pendingDelete == nil {
					pendingDelete = map[string]job{}
				}
				k := jobKey(j)
				if _, ok := pendingDelete[k]; !ok {
					pendingDelete[k] = j
				}
				pendingDirtyMu.Unlock()
				continue
			}
			log.Printf("embedqueue: clear pending %s: %v", j.file, err)
		}
	}

	if err := tx.Commit(); err != nil {
		if db.IsDBLocked(err) {
			db.NoteDBLock()
			requeuePendingFlush(upserts, deletes)
			return
		}
		log.Printf("embedqueue: pending batch commit: %v", err)
		requeuePendingFlush(upserts, deletes)
		return
	}
	if !locked {
		db.NoteDBOK()
	}
}

func requeuePendingFlush(upserts map[string]pendingDirtyRow, deletes map[string]job) {
	pendingDirtyMu.Lock()
	defer pendingDirtyMu.Unlock()
	if pendingDirty == nil {
		pendingDirty = map[string]pendingDirtyRow{}
	}
	if pendingDelete == nil {
		pendingDelete = map[string]job{}
	}
	for k, row := range upserts {
		if _, ok := pendingDirty[k]; !ok {
			pendingDirty[k] = row
		}
	}
	for k, j := range deletes {
		if _, ok := pendingDelete[k]; !ok {
			pendingDelete[k] = j
		}
	}
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
	type row struct{ file, projectPath string }
	var loadedRows []row
	for rows.Next() {
		var r row
		if err := rows.Scan(&r.file, &r.projectPath); err != nil {
			continue
		}
		loadedRows = append(loadedRows, r)
	}
	rows.Close()
	loaded := 0
	pendingMu.Lock()
	if pending == nil {
		pending = map[string]job{}
	}
	for _, r := range loadedRows {
		if indexer.ShouldSkipEmbed(r.file) {
			continue
		}
		k := jobKey(job{file: r.file, projectPath: r.projectPath})
		if _, ok := pending[k]; ok {
			continue
		}
		pending[k] = job{file: r.file, projectPath: r.projectPath}
		loaded++
	}
	n := len(pending)
	pendingMu.Unlock()
	trackPendingPeak(n)
	if loaded > 0 {
		log.Printf("embedqueue: loaded %d pending from DB", loaded)
	}
	return loaded
}
