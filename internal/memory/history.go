package memory

import (
	"database/sql"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	historySelect = `SELECT sm.ref, sm.kind, sm.scope, sm.session_id, sm.project_path, sm.subject, sm.predicate, sm.object, sm.rule,
		sm.valid_from, sm.valid_until, sm.superseded_by, sm.source_ref, sm.token_est, sm.access_count, sm.last_accessed_at, sm.created_at,
		sm.invalidated_reason, sm.world_valid_at, sm.world_invalid_at, sm.expires_at
		FROM structured_memory sm`
)

// HistoryEntry is a version of a memory entry with its invalidation reason and timestamps.
type HistoryEntry struct {
	Ref               string `json:"ref"`
	Kind              Kind   `json:"kind"`
	Scope             Scope  `json:"scope"`
	SessionID         string `json:"session_id,omitempty"`
	ProjectPath       string `json:"project_path,omitempty"`
	Subject           string `json:"subject,omitempty"`
	Predicate         string `json:"predicate,omitempty"`
	Object            string `json:"object,omitempty"`
	Rule              string `json:"rule,omitempty"`
	ValidFrom         string `json:"valid_from,omitempty"`
	ValidUntil        string `json:"valid_until,omitempty"`
	SupersededBy      string `json:"superseded_by,omitempty"`
	SourceRef         string `json:"source_ref,omitempty"`
	TokenEst          int    `json:"token_est,omitempty"`
	AccessCount       int    `json:"access_count,omitempty"`
	LastAccessedAt    string `json:"last_accessed_at,omitempty"`
	CreatedAt         string `json:"created_at,omitempty"`
	InvalidatedReason string `json:"invalidated_reason,omitempty"`
	WorldValidAt      string `json:"world_valid_at,omitempty"`
	WorldInvalidAt    string `json:"world_invalid_at,omitempty"`
	ExpiresAt         string `json:"expires_at,omitempty"`
}

// HistoryResult is the result of a recall_memory history query.
type HistoryResult struct {
	Entries    []HistoryEntry `json:"entries"`
	Formatted  string         `json:"formatted"`
	TokensUsed int            `json:"tokens_used"`
}

// HistoryInput configures a recall_memory history query.
type HistoryInput struct {
	Ref          string // optional: single entry by ref
	Query        string // optional: query string for subject/predicate/object/rule
	IncludeWorld bool   // include world-time columns
}

// History returns the full version chain for a memory entry or entries matching
// a subject+predicate, with reasons and timestamps. It is used by
// recall_memory(history=true, ref=...) and recall_memory(history=true, query=...).
func History(in HistoryInput) (*HistoryResult, error) {
	if in.Ref != "" {
		return historyByRef(in.Ref, in.IncludeWorld)
	}
	if in.Query != "" {
		return historyByQuery(in.Query, in.IncludeWorld)
	}
	return nil, errs.NewCode(errs.CodeInvalidInput, "history requires either a ref or a query")
}

func historyByRef(ref string, includeWorld bool) (*HistoryResult, error) {
	// MEM-3: follow the version chain in both directions from the given ref.
	// Backward: entries whose superseded_by points at this ref (or any ref in the chain).
	// Forward: the entry this ref points to via superseded_by, and so on.
	seen := map[string]bool{ref: true}
	queue := []string{ref}
	var all []HistoryEntry

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		e, err := historyEntryByRef(cur, includeWorld)
		if err != nil {
			continue
		}
		all = append(all, e)
		// Forward: this entry was superseded by e.SupersededBy.
		if e.SupersededBy != "" && !seen[e.SupersededBy] {
			seen[e.SupersededBy] = true
			queue = append(queue, e.SupersededBy)
		}
		// Backward: other entries that were superseded by this one.
		back, err := historyEntriesSupersededBy(cur, includeWorld)
		if err != nil {
			continue
		}
		for _, b := range back {
			if !seen[b.Ref] {
				seen[b.Ref] = true
				queue = append(queue, b.Ref)
			}
		}
	}

	if len(all) == 0 {
		return nil, errs.NewCode(errs.CodeNotFound, "no history entries found for ref %q", ref)
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].CreatedAt < all[j].CreatedAt
	})
	return &HistoryResult{Entries: all}, nil
}

func historyEntryByRef(ref string, includeWorld bool) (HistoryEntry, error) {
	q := historySelect
	if !includeWorld {
		q += ` WHERE sm.ref = ? AND (sm.world_valid_at IS NULL OR sm.world_valid_at = '')`
	} else {
		q += ` WHERE sm.ref = ?`
	}
	q += ` ORDER BY sm.created_at ASC LIMIT 1`
	row := db.ContextDB.QueryRow(q, ref)
	var e HistoryEntry
	var validUntil, invalidatedReason, expiresAt sql.NullString
	var worldValidAt, worldInvalidAt sql.NullString
	err := row.Scan(&e.Ref, &e.Kind, &e.Scope, &e.SessionID, &e.ProjectPath,
		&e.Subject, &e.Predicate, &e.Object, &e.Rule,
		&e.ValidFrom, &validUntil, &e.SupersededBy, &e.SourceRef, &e.TokenEst,
		&e.AccessCount, &e.LastAccessedAt, &e.CreatedAt,
		&invalidatedReason, &worldValidAt, &worldInvalidAt, &expiresAt)
	if err != nil {
		return e, err
	}
	e.ValidUntil = validUntil.String
	e.InvalidatedReason = invalidatedReason.String
	e.WorldValidAt = worldValidAt.String
	e.WorldInvalidAt = worldInvalidAt.String
	e.ExpiresAt = expiresAt.String
	return e, nil
}

func historyEntriesSupersededBy(ref string, includeWorld bool) ([]HistoryEntry, error) {
	q := historySelect
	if !includeWorld {
		q += ` WHERE sm.superseded_by = ? AND (sm.world_valid_at IS NULL OR sm.world_valid_at = '')`
	} else {
		q += ` WHERE sm.superseded_by = ?`
	}
	q += ` ORDER BY sm.created_at ASC`
	rows, err := db.ContextDB.Query(q, ref)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []HistoryEntry
	for rows.Next() {
		var e HistoryEntry
		var validUntil, invalidatedReason, expiresAt sql.NullString
		var worldValidAt, worldInvalidAt sql.NullString
		err := rows.Scan(&e.Ref, &e.Kind, &e.Scope, &e.SessionID, &e.ProjectPath,
			&e.Subject, &e.Predicate, &e.Object, &e.Rule,
			&e.ValidFrom, &validUntil, &e.SupersededBy, &e.SourceRef, &e.TokenEst,
			&e.AccessCount, &e.LastAccessedAt, &e.CreatedAt,
			&invalidatedReason, &worldValidAt, &worldInvalidAt, &expiresAt)
		if err != nil {
			return out, err
		}
		e.ValidUntil = validUntil.String
		e.InvalidatedReason = invalidatedReason.String
		e.WorldValidAt = worldValidAt.String
		e.WorldInvalidAt = worldInvalidAt.String
		e.ExpiresAt = expiresAt.String
		out = append(out, e)
	}
	return out, rows.Err()
}

func historyByQuery(query string, includeWorld bool) (*HistoryResult, error) {
	q := historySelect
	var args []any
	var where []string
	if !includeWorld {
		where = append(where, "(sm.world_valid_at IS NULL OR sm.world_valid_at = '')")
	}
	like := "%" + strings.ReplaceAll(query, "%", "\\%") + "%"
	content := make([]string, 0, 4)
	for _, col := range []string{"sm.subject", "sm.predicate", "sm.object", "sm.rule"} {
		content = append(content, col+" LIKE ?")
		args = append(args, like)
	}
	where = append(where, "("+strings.Join(content, " OR ")+")")
	q += " WHERE " + strings.Join(where, " AND ") + " ORDER BY sm.ref LIMIT 50"

	rows, err := db.ContextDB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []HistoryEntry
	for rows.Next() {
		var e HistoryEntry
		var validUntil, invalidatedReason, expiresAt sql.NullString
		var worldValidAt, worldInvalidAt sql.NullString
		err := rows.Scan(&e.Ref, &e.Kind, &e.Scope, &e.SessionID, &e.ProjectPath,
			&e.Subject, &e.Predicate, &e.Object, &e.Rule,
			&e.ValidFrom, &validUntil, &e.SupersededBy, &e.SourceRef, &e.TokenEst,
			&e.AccessCount, &e.LastAccessedAt, &e.CreatedAt,
			&invalidatedReason, &worldValidAt, &worldInvalidAt, &expiresAt)
		if err != nil {
			return nil, err
		}
		e.ValidUntil = validUntil.String
		e.InvalidatedReason = invalidatedReason.String
		e.WorldValidAt = worldValidAt.String
		e.WorldInvalidAt = worldInvalidAt.String
		e.ExpiresAt = expiresAt.String
		entries = append(entries, e)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return &HistoryResult{Entries: entries}, nil
}
