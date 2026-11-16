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

// invalidateConflicting supersedes every currently active entry of kind with the same
// subject, predicate and scope, each marked with ReasonContradiction. It is
// called with factSupersessionMu held by the caller. The new ref is excluded.
func invalidateConflicting(kind Kind, scope Scope, sessionID, projectPath, subject, predicate, newRef string) ([]string, error) {
	if kind == "" {
		kind = KindFact
	}
	q := selectActiveFactRefsQuery
	args := []any{string(kind), subject, predicate}
	q, args = appendScopeFilter(q, args, scope, sessionID, projectPath)
	return invalidateMatchingRefs(q, args, ReasonContradiction, newRef)
}

// invalidateTargetChain supersedes every currently active config_change entry with
// the same target and scope, each marked with ReasonSuperseded and linked via
// superseded_by = newRef. Called with factSupersessionMu held by the caller.
func invalidateTargetChain(scope Scope, sessionID, projectPath, target, newRef string) ([]string, error) {
	q := selectActiveTargetRefsQuery
	args := []any{target}
	q, args = appendScopeFilter(q, args, scope, sessionID, projectPath)
	return invalidateMatchingRefs(q, args, ReasonSuperseded, newRef)
}

func appendScopeFilter(q string, args []any, scope Scope, sessionID, projectPath string) (string, []any) {
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
	return q, args
}

func invalidateMatchingRefs(q string, args []any, reason, newRef string) ([]string, error) {
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
		if err := invalidate(db.ContextDB, ref, reason, newRef); err != nil {
			logger.Warn("Failed to invalidate superseded memory entry", "ref", ref, "reason", reason, "error", err)
		}
	}
	return refs, nil
}
