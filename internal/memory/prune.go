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
func PruneSuperseded(maxAgeDays int) (int64, error) {
	if maxAgeDays <= 0 {
		maxAgeDays = 90
	}
	cutoff := db.SQLTime(time.Now().UTC().AddDate(0, 0, -maxAgeDays))
	rows, err := db.ContextDB.Query(selectExpiredEntryRefsQuery, cutoff)
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
	// The vectors live in index.db. Delete them first and stop if that fails (a
	// WAL quiesce gates index writes): the rows stay for the next prune, rather
	// than going away and leaving their vectors orphaned.
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
	res, err := db.ContextDB.Exec(deleteExpiredEntriesQuery, cutoff)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		logger.Info("Pruned superseded structured memory", "pruned", n, "max_age_days", maxAgeDays)
	}
	return n, nil
}
