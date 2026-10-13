package memory

import (
	"database/sql"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// invalidate marks the row with ref as invalid: sets valid_until to now,
// writes superseded_by and invalidated_reason, and removes its FTS and vector
// indices. Vector deletion happens outside the tx via the vector cache.
//
// Invalidated rows are never deleted; they stay queryable via history for the
// archive period (MEM-2).
func invalidate(dbx *sql.DB, ref, reason, supersededBy string) error {
	_, err := dbx.Exec(supersedeEntryQuery, supersededBy, reason, ref)
	if err != nil {
		return err
	}
	deleteFTS(ref)
	search.Cache.DeleteRefs("memory", []string{memoryVectorKey(ref)})
	return nil
}

// invalidateConflicting supersedes every currently active fact with the same
// subject, predicate and scope, each marked with reasonContradiction. It is
// called with factSupersessionMu held by the caller. The new ref is excluded.
func invalidateConflicting(scope Scope, sessionID, projectPath, subject, predicate, newRef string) ([]string, error) {
	q := selectActiveFactRefsQuery
	args := []any{subject, predicate}
	switch scope {
	case ScopeSession:
		q += andSessionScopeClause
		args = append(args, sessionID)
	case ScopeProject:
		q += andProjectScopeClause
		args = append(args, projectPath)
	case ScopeGlobal:
		q += andGlobalScopeClause
	}
	rows, err := db.ContextDB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var refs []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" && ref != newRef {
			refs = append(refs, ref)
		}
	}
	for _, ref := range refs {
		invalidate(db.ContextDB, ref, ReasonContradiction, newRef)
	}
	return refs, nil
}
