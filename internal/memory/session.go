package memory

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	selectActiveSessionEntriesQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE scope = 'session' AND session_id = ? AND (valid_until IS NULL OR valid_until = '')
		ORDER BY created_at, ref`
	selectSessionEntryRefsQuery = `SELECT ref FROM structured_memory WHERE scope = 'session' AND session_id = ?`
	deleteSessionEntriesQuery   = `DELETE FROM structured_memory WHERE scope = 'session' AND session_id = ?`
)

// ActiveForSession returns sessionID's current (not superseded or forgotten) session-scoped
// entries, oldest first, without recording an access. A handoff snapshot copies them.
func ActiveForSession(sessionID string) ([]Entry, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "session_id required")
	}
	entries, err := queryEntries(selectActiveSessionEntriesQuery, sessionID)
	if err != nil {
		return nil, errs.WrapMessage("failed to read session memory", err, "session_id", sessionID)
	}
	return entries, nil
}

// DeleteSession permanently deletes every session-scoped entry of sessionID, current or not,
// with its FTS row and vector, and returns how many rows were deleted. Unlike Forget it leaves
// no history: tree expiry uses it for child sessions, which nothing can recall afterwards.
func DeleteSession(sessionID string) (int, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return 0, errs.NewCode(errs.CodeInvalidInput, "session_id required")
	}
	rows, err := db.ContextDB.Query(selectSessionEntryRefsQuery, sessionID)
	if err != nil {
		return 0, errs.WrapMessage("failed to list session memory", err, "session_id", sessionID)
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
	// Vectors first, as PruneSuperseded does: if index writes are gated the rows stay for a
	// retry instead of leaving orphaned vectors behind.
	keys := make([]string, len(refs))
	for i, ref := range refs {
		keys[i] = memoryVectorKey(ref)
	}
	if err := search.Cache.DeleteRefs("memory", keys); err != nil {
		return 0, errs.WrapMessage("failed to delete session memory vectors", err, "session_id", sessionID)
	}
	for _, ref := range refs {
		deleteFTS(ref)
	}
	res, err := db.ContextDB.Exec(deleteSessionEntriesQuery, sessionID)
	if err != nil {
		return 0, errs.WrapMessage("failed to delete session memory", err, "session_id", sessionID)
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}
