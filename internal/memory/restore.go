package memory

import (
	"database/sql"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// RestoreInput configures a restore action in forget_memory.
type RestoreInput struct {
	Refs []string
}

// RestoreResult describes the outcome of restoring superseded facts.
type RestoreResult struct {
	RestoredRefs []string `json:"restored_refs"`
	Conflicts    []string `json:"conflicts,omitempty"`
}

// Restore makes a superseded/forgotten fact active again, and invalidates the fact
// that superseded it, unless that fact has been superseded again, in which case it reports a conflict.
func Restore(in RestoreInput) (*RestoreResult, error) {
	var restored []string
	var conflicts []string

	for _, ref := range in.Refs {
		ref = strings.TrimSpace(ref)
		if ref == "" {
			continue
		}

		tx, err := db.ContextDB.Begin()
		if err != nil {
			return nil, err
		}

		// 1. Get the entry to restore
		var supersededBy string
		var validUntil string
		var subject, predicate, object, rule string
		err = tx.QueryRow(`SELECT superseded_by, valid_until, subject, predicate FROM structured_memory WHERE ref = ?`, ref).Scan(&supersededBy, &validUntil, &subject, &predicate)
		if err == sql.ErrNoRows {
			tx.Rollback()
			conflicts = append(conflicts, ref+": not found")
			continue
		} else if err != nil {
			tx.Rollback()
			return nil, err
		}

		if validUntil == "" {
			tx.Rollback()
			// Already active, skip or treat as success
			restored = append(restored, ref)
			continue
		}

		// 2. Check if the fact that superseded it is still the current active fact for this subject+predicate
		if supersededBy != "" {
			var currValidUntil string
			err = tx.QueryRow(`SELECT valid_until FROM structured_memory WHERE ref = ?`, supersededBy).Scan(&currValidUntil)
			if err == sql.ErrNoRows {
				// Superseding entry no longer exists; nothing to invalidate, restore as-is.
			} else if err == nil && currValidUntil != "" {
				// The superseding fact is still active — conflict!
				tx.Rollback()
				conflicts = append(conflicts, ref+": superseding entry "+supersededBy+" is still active (conflict)")
				continue
			}
		}

		// 3. Perform restoration: clear valid_until and invalidated_reason on target ref
		_, err = tx.Exec(`UPDATE structured_memory SET valid_until = NULL, invalidated_reason = NULL, access_count = access_count + 1 WHERE ref = ?`, ref)
		if err != nil {
			tx.Rollback()
			return nil, err
		}

		// Re-insert the restored entry into FTS so it becomes searchable again.
		if err := tx.QueryRow(`SELECT subject, predicate, object, rule FROM structured_memory WHERE ref = ?`, ref).Scan(&subject, &predicate, &object, &rule); err != nil {
			tx.Rollback()
			return nil, err
		}
		indexFTS(ref, subject, predicate, object, rule)

		if supersededBy != "" {
			_, err = tx.Exec(`UPDATE structured_memory SET valid_until = datetime('now'), invalidated_reason = ? WHERE ref = ?`, ReasonRestored, supersededBy)
			if err != nil {
				tx.Rollback()
				return nil, err
			}
			deleteFTS(supersededBy)
		}

		if err := tx.Commit(); err != nil {
			return nil, err
		}
		restored = append(restored, ref)
	}

	return &RestoreResult{RestoredRefs: restored, Conflicts: conflicts}, nil
}