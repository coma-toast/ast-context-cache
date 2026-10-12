package memory

import (
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	selectExpiredEntryRefsQuery = `SELECT ref FROM structured_memory WHERE valid_until IS NOT NULL AND valid_until != '' AND valid_until < ?`
	deleteExpiredEntriesQuery   = `DELETE FROM structured_memory WHERE valid_until IS NOT NULL AND valid_until != '' AND valid_until < ?`
)

// PruneSuperseded permanently deletes structured_memory rows that were
// invalidated (superseded by a newer fact, or explicitly forgotten) more than
// maxAgeDays ago. Superseded facts are soft-deleted via valid_until so
// recall can still reason about "what used to be true" for a while, but
// nothing ever purged old rows afterward — unlike contextnotes (a
// token/count quota, see limits.go) and the queries table (age-based
// retention, see db.RunQueryRetention), structured_memory grew unbounded.
// Per MEM-2: the dashboard 90-day prune must not delete invalidated rows
// (rows with invalidated_reason set). Those are handled by the daily
// PruneArchived job with a configurable window keyed on valid_until.
func PruneSuperseded(maxAgeDays int) (int64, error) {
	if maxAgeDays <= 0 {
		maxAgeDays = 90
	}
	cutoff := db.SQLTime(time.Now().UTC().AddDate(0, 0, -maxAgeDays))
	// Only delete rows that have valid_until but NO invalidated_reason.
	// Rows with invalidated_reason are handled by PruneArchived with its
	// configurable window keyed on valid_until.
	rows, err := db.ContextDB.Query(
		`SELECT ref FROM structured_memory
		 WHERE valid_until IS NOT NULL AND valid_until != ''
		   AND (invalidated_reason IS NULL OR invalidated_reason = '')
		   AND valid_until < ?`,
		cutoff,
	)
	if err != nil {
		return 0, err
	}
	var refs []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" {
			refs = append(refs, ref)
		}
	}
	rows.Close()
	if len(refs) == 0 {
		return 0, nil
	}
	keys := make([]string, len(refs))
	for i, ref := range refs {
		keys[i] = memoryVectorKey(ref)
	}
	if err := search.Cache.DeleteRefs("memory", keys); err != nil {
		return 0, errs.WrapMessage("failed to delete memory vectors", err)
	}
	for _, ref := range refs {
		deleteFTS(ref)
	}
	res, err := db.ContextDB.Exec(
		`DELETE FROM structured_memory
		 WHERE valid_until IS NOT NULL AND valid_until != ''
		   AND (invalidated_reason IS NULL OR invalidated_reason = '')
		   AND valid_until < ?`,
		cutoff,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		logger.Info("Pruned superseded structured memory", "pruned", n, "max_age_days", maxAgeDays)
	}
	return n, nil
}

// PruneArchived permanently deletes structured_memory rows archived (invalidated)
// longer than the queryable window. The window is a setting (default 365 days).
// It is the daily job from MEM-2: invalidated rows stay readable via history
// for the queryable window, after which they are purged entirely. Keyed on
// invalidation time (valid_until), not world_invalid_at.
func PruneArchived(queryableWindowDays int) (int64, error) {
	if queryableWindowDays <= 0 {
		queryableWindowDays = db.SettingInt("memory_archive_window_days", "", 365)
	}
	cutoff := db.SQLTime(time.Now().UTC().AddDate(0, 0, -queryableWindowDays))
	rows, err := db.ContextDB.Query(
		`SELECT ref FROM structured_memory
		 WHERE valid_until IS NOT NULL AND valid_until != ''
		   AND invalidated_reason IS NOT NULL AND invalidated_reason != ''
		   AND valid_until < ?`,
		cutoff,
	)
	if err != nil {
		return 0, err
	}
	var refs []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" {
			refs = append(refs, ref)
		}
	}
	rows.Close()
	if len(refs) == 0 {
		return 0, nil
	}
	keys := make([]string, len(refs))
	for i, ref := range refs {
		keys[i] = memoryVectorKey(ref)
	}
	if err := search.Cache.DeleteRefs("memory", keys); err != nil {
		return 0, errs.WrapMessage("failed to delete memory vectors", err)
	}
	for _, ref := range refs {
		deleteFTS(ref)
	}
	res, err := db.ContextDB.Exec(
		`DELETE FROM structured_memory
		 WHERE valid_until IS NOT NULL AND valid_until != ''
		   AND invalidated_reason IS NOT NULL AND invalidated_reason != ''
		   AND valid_until < ?`,
		cutoff,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		logger.Info("Pruned archived structured memory", "pruned", n, "window_days", queryableWindowDays)
	}
	return n, nil
}

// RunDailyPrune runs the daily MEM-2 job: call PruneArchived with the configured window.
// This job should be started once at server startup and run every 24 hours.
func RunDailyPrune() {
	// The PruneArchived function now reads the window from settings
	// (memory_archive_window_days, default 365) via db.SettingInt.
	if _, err := PruneArchived(0); err != nil {
		logger.Error("Failed to run daily memory prune", "error", err)
	}
}

// RunExpiryJob runs the periodic MEM-8 job: invalidate facts that have passed their expires_at.
// This job should be started once at server startup and run periodically (e.g. hourly).
// It sets valid_until to now and invalidated_reason to "ttl" for any fact where
// expires_at is set and in the past. Rows without expires_at are ignored.
// After DB update, it also deletes the FTS and vector entries for consistency.
// This complements the recall-time applyTTLExpiry which only affects output.
func RunExpiryJob() {
	now := db.SQLTime(time.Now().UTC())
	res, err := db.ContextDB.Exec(
		`UPDATE structured_memory
		 SET valid_until = ?, invalidated_reason = ?
		 WHERE kind = 'fact'
		   AND expires_at IS NOT NULL AND expires_at != ''
		   AND expires_at < ?`,
		now, ReasonTTL, now,
	)
	if err != nil {
		logger.Error("Failed to run memory expiry job", "error", err)
		return
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return
	}
	// Need to get the refs that were updated to delete their vectors and FTS entries.
	rows, err := db.ContextDB.Query(
		`SELECT ref FROM structured_memory
		 WHERE valid_until = ? AND invalidated_reason = ? AND expires_at < ?`,
		now, ReasonTTL, now,
	)
	if err != nil {
		logger.Error("Failed to fetch expired refs for vector/FTs cleanup", "error", err)
		return
	}
	var refs []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" {
			refs = append(refs, ref)
		}
	}
	rows.Close()
	if len(refs) == 0 {
		return
	}
	// Delete vectors first (to avoid orphaned vectors if FTS deletion fails)
	keys := make([]string, len(refs))
	for i, ref := range refs {
		keys[i] = memoryVectorKey(ref)
	}
	if err := search.Cache.DeleteRefs("memory", keys); err != nil {
		logger.Error("Failed to delete memory vectors for expiry", "error", err)
	}
	for _, ref := range refs {
		deleteFTS(ref)
	}
	logger.Info("Expired structured memory invalidated via TTL", "count", n)
}
