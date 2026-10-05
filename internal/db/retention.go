package db

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

const deleteExpiredQueriesQuery = "DELETE FROM queries WHERE timestamp < ?"

// RunQueryRetention deletes old rows from the queries table per dashboard settings.
// Returns rows deleted, 0 when nothing to delete, or -1 when SQLite is busy (caller should retry).
func RunQueryRetention() int64 {
	if strings.ToLower(strings.TrimSpace(GetSetting("query_retention_enabled", "true"))) == "false" {
		return 0
	}
	if ShouldThrottleHeavyWork() && dbLockStreak.Load() > 0 {
		return -1
	}
	maxDays, _ := strconv.Atoi(strings.TrimSpace(GetSetting("query_retention_max_age_days", "90")))
	if maxDays <= 0 {
		maxDays = 90
	}
	cutoff := time.Now().AddDate(0, 0, -maxDays).Format("2006-01-02") + "T00:00:00"
	res, err := DB.Exec(deleteExpiredQueriesQuery, cutoff)
	if err != nil {
		if IsDBLocked(err) {
			NoteDBLock()
			return -1
		}
		logger.Warn("Failed to delete expired queries", "error", err)
		return 0
	}
	NoteDBOK()
	n, _ := res.RowsAffected()
	if n > 0 {
		logger.Info("Deleted expired queries", "rows", n, "max_age_days", maxDays)
		if busy, frames, ckpt, err := CheckpointWAL(true); err == nil {
			logger.Info("Checkpointed WAL after query retention", "busy", busy, "wal_frames", frames, "checkpointed", ckpt)
		}
	}
	meta, _ := json.Marshal(map[string]any{
		"at":       time.Now().Format(time.RFC3339),
		"deleted":  n,
		"max_days": maxDays,
	})
	_ = SetSetting("query_retention_last_run", string(meta))
	return n
}
