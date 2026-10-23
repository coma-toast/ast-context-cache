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

// PruneSuperseded was the dashboard 90-day prune for superseded memory.
// Per MEM-2: the 90-day prune must not delete any invalidated rows
// (any row with valid_until set). All invalidated rows are handled by
// PruneArchived with its 365-day window keyed on valid_until.
// This function is now a no-op and retained for API compatibility.
func PruneSuperseded(maxAgeDays int) (int64, error) {
	return 0, nil
}

// PruneArchived permanently deletes structured_memory rows archived (invalidated)
// longer than the queryable window. The window is a setting (default 365 days).
// It is the daily job from MEM-2: invalidated rows stay readable via history
// for the queryable window, after which they are purged entirely. Keyed on
// invalidation time (valid_until), not world_invalid_at.
// MEM-2: any row with valid_until set is treated as invalidated (rows invalidated
// before the migration carry no invalidated_reason, so we don't filter on it), so
// the 90-day prune never deletes them.
func PruneArchived(queryableWindowDays int) (int64, error) {
	if queryableWindowDays <= 0 {
		queryableWindowDays = db.SettingInt("memory_archive_window_days", "", 365)
	}
	cutoff := db.SQLTime(time.Now().UTC().AddDate(0, 0, -queryableWindowDays))
	rows, err := db.ContextDB.Query(
		`SELECT ref FROM structured_memory
		 WHERE valid_until IS NOT NULL AND valid_until != ''
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
// Gated by feature_memory_v2 (MEM-11).
func RunExpiryJob() {
	if !v2Enabled() {
		return
	}
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

// RunStaleAnchorJob runs the periodic MEM-1 job: invalidate facts whose world-time
// anchor has passed (world_invalid_at is set and earlier than now). A fact whose
// world_invalid_at is in the past is no longer true in the world, so it is
// invalidated with reason "stale_anchor". Rows without world_invalid_at are ignored.
func RunStaleAnchorJob() {
	now := db.SQLTime(time.Now().UTC())
	res, err := db.ContextDB.Exec(
		`UPDATE structured_memory
		 SET valid_until = ?, invalidated_reason = ?
		 WHERE kind = 'fact'
		   AND (valid_until IS NULL OR valid_until = '')
		   AND world_invalid_at IS NOT NULL AND world_invalid_at != ''
		   AND world_invalid_at < ?`,
		now, ReasonStaleAnchor, now,
	)
	if err != nil {
		logger.Error("Failed to run stale anchor job", "error", err)
		return
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return
	}
	rows, err := db.ContextDB.Query(
		`SELECT ref FROM structured_memory
		 WHERE valid_until = ? AND invalidated_reason = ?`,
		now, ReasonStaleAnchor,
	)
	if err != nil {
		logger.Error("Failed to fetch stale anchor refs for cleanup", "error", err)
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
	keys := make([]string, len(refs))
	for i, ref := range refs {
		keys[i] = memoryVectorKey(ref)
	}
	if err := search.Cache.DeleteRefs("memory", keys); err != nil {
		logger.Error("Failed to delete memory vectors for stale anchors", "error", err)
	}
	for _, ref := range refs {
		deleteFTS(ref)
	}
	logger.Info("Structured memory invalidated via stale world anchors", "count", n)
}

// RejectQuarantined invalidates a quarantined entry with reason "quarantine" (MEM-1,
// TR-3). Quarantined entries are rejected from the dashboard review queue; rejecting
// must invalidate the row (never delete) so history keeps explaining what happened.
func RejectQuarantined(ref string) error {
	if ref == "" {
		return errs.NewCode(errs.CodeInvalidInput, "ref required")
	}
	res, err := db.ContextDB.Exec(
		`UPDATE structured_memory
		 SET valid_until = datetime('now'), invalidated_reason = ?
		 WHERE ref = ? AND quarantined = 1 AND (valid_until IS NULL OR valid_until = '')`,
		ReasonQuarantine, ref,
	)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errs.NewCode(errs.CodeNotFound, "no active quarantined entry with ref %q", ref)
	}
	deleteFTS(ref)
	if err := search.Cache.DeleteRefs("memory", []string{memoryVectorKey(ref)}); err != nil {
		logger.Warn("Failed to delete memory vector for quarantined ref", "ref", ref, "error", err)
	}
	logger.Info("Quarantined entry rejected", "ref", ref)
	return nil
}
