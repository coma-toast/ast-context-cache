package memory

import (
	"database/sql"
	"errors"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func historySelectSQL() string {
	return "SELECT " + entrySelectColsAliased("sm", true) + " FROM structured_memory sm"
}

// HistoryEntry is a version of a memory entry with its invalidation reason and timestamps.
type HistoryEntry struct {
	Ref               string   `json:"ref"`
	Kind              Kind     `json:"kind"`
	Scope             Scope    `json:"scope"`
	SessionID         string   `json:"session_id,omitempty"`
	ProjectPath       string   `json:"project_path,omitempty"`
	Subject           string   `json:"subject,omitempty"`
	Predicate         string   `json:"predicate,omitempty"`
	Object            string   `json:"object,omitempty"`
	Rule              string   `json:"rule,omitempty"`
	ValidFrom         string   `json:"valid_from,omitempty"`
	ValidUntil        string   `json:"valid_until,omitempty"`
	SupersededBy      string   `json:"superseded_by,omitempty"`
	SourceRef         string   `json:"source_ref,omitempty"`
	TokenEst          int      `json:"token_est,omitempty"`
	AccessCount       int      `json:"access_count,omitempty"`
	LastAccessedAt    string   `json:"last_accessed_at,omitempty"`
	CreatedAt         string   `json:"created_at,omitempty"`
	InvalidatedReason string   `json:"invalidated_reason,omitempty"`
	WorldValidAt      string   `json:"world_valid_at,omitempty"`
	WorldInvalidAt    string   `json:"world_invalid_at,omitempty"`
	ExpiresAt         string   `json:"expires_at,omitempty"`
	Quarantined       int      `json:"quarantined,omitempty"`
	Title             string   `json:"title,omitempty"`
	Body              string   `json:"body,omitempty"`
	Tags              []string `json:"tags,omitempty"`
	Actor             string   `json:"actor,omitempty"`
	Category          string   `json:"category,omitempty"`
	Severity          string   `json:"severity,omitempty"`
	Target            string   `json:"target,omitempty"`
	BeforeValue       string   `json:"before_value,omitempty"`
	AfterValue        string   `json:"after_value,omitempty"`
	Reason            string   `json:"reason,omitempty"`
	OccurredAt        string   `json:"occurred_at,omitempty"`
	Verified          bool     `json:"verified,omitempty"`
	VerifiedAt        string   `json:"verified_at,omitempty"`
	RedactedFields    []string `json:"redacted_fields,omitempty"`
}

// HistoryResult is the result of a recall_memory history query.
type HistoryResult struct {
	Entries    []HistoryEntry `json:"entries"`
	Formatted  string         `json:"formatted"`
	TokensUsed int            `json:"tokens_used"`
}

// HistoryInput configures a recall_memory history query.
type HistoryInput struct {
	Ref                 string // optional: single entry by ref
	Query               string // optional: query string for subject/predicate/object/rule/title/target/reason
	Subject             string // optional: exact subject (MEM-3)
	Predicate           string // optional: exact predicate (MEM-3)
	Scope               Scope  // optional: scope filter for query mode
	SessionID           string // optional: session filter for query mode
	ProjectPath         string // optional: project filter for query mode
	IncludeRepoSiblings bool   // optional: include repo siblings when ProjectPath is set
	IncludeWorld        bool   // include world-time columns
}

// History returns the full version chain for a memory entry or entries matching
// a query, with reasons and timestamps. It is used by
// recall_memory(history=true, ref=...) and recall_memory(history=true, query=...).
func History(in HistoryInput) (*HistoryResult, error) {
	if in.Ref != "" {
		return historyByRef(in.Ref, in.IncludeWorld)
	}
	if in.Query != "" {
		return historyByQuery(in)
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
			if errors.Is(err, sql.ErrNoRows) {
				continue
			}
			return nil, err
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
			return nil, err
		}
		for _, b := range back {
			if !seen[b.Ref] {
				seen[b.Ref] = true
				queue = append(queue, b.Ref)
			}
		}
	}

	if len(all) == 0 {
		return nil, errs.NewCode(errs.CodeNotFound, "no history entries found for ref "+ref)
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].CreatedAt < all[j].CreatedAt
	})
	return &HistoryResult{Entries: all}, nil
}

func historyEntryByRef(ref string, includeWorld bool) (HistoryEntry, error) {
	q := historySelectSQL()
	if !includeWorld {
		q += ` WHERE sm.ref = ? AND (sm.world_valid_at IS NULL OR sm.world_valid_at = '')`
	} else {
		q += ` WHERE sm.ref = ?`
	}
	q += ` ORDER BY sm.created_at ASC LIMIT 1`
	row := db.ContextDB.QueryRow(q, ref)
	e, err := scanEntry(row)
	if err != nil {
		return HistoryEntry{}, err
	}
	return entryToHistoryEntry(e), nil
}

func historyEntriesSupersededBy(ref string, includeWorld bool) ([]HistoryEntry, error) {
	q := historySelectSQL()
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
		e, err := scanEntry(rows)
		if err != nil {
			return out, err
		}
		out = append(out, entryToHistoryEntry(e))
	}
	return out, rows.Err()
}

func entryToHistoryEntry(e Entry) HistoryEntry {
	return HistoryEntry{
		Ref:               e.Ref,
		Kind:              e.Kind,
		Scope:             e.Scope,
		SessionID:         e.SessionID,
		ProjectPath:       e.ProjectPath,
		Subject:           e.Subject,
		Predicate:         e.Predicate,
		Object:            e.Object,
		Rule:              e.Rule,
		ValidFrom:         e.ValidFrom,
		ValidUntil:        e.ValidUntil,
		SupersededBy:      e.SupersededBy,
		SourceRef:         e.SourceRef,
		TokenEst:          e.TokenEst,
		AccessCount:       e.AccessCount,
		LastAccessedAt:    e.LastAccessedAt,
		CreatedAt:         e.CreatedAt,
		InvalidatedReason: e.InvalidatedReason,
		WorldValidAt:      e.WorldValidAt,
		WorldInvalidAt:    e.WorldInvalidAt,
		ExpiresAt:         e.ExpiresAt,
		Quarantined:       e.Quarantined,
		Title:             e.Title,
		Body:              e.Body,
		Tags:              e.Tags,
		Actor:             e.Actor,
		Category:          e.Category,
		Severity:          e.Severity,
		Target:            e.Target,
		BeforeValue:       e.BeforeValue,
		AfterValue:        e.AfterValue,
		Reason:            e.Reason,
		OccurredAt:        e.OccurredAt,
		Verified:          e.Verified,
		VerifiedAt:        e.VerifiedAt,
		RedactedFields:    e.RedactedFields,
	}
}

func historyByQuery(in HistoryInput) (*HistoryResult, error) {
	q := historySelectSQL()
	var args []any
	var where []string
	if !in.IncludeWorld {
		where = append(where, "(sm.world_valid_at IS NULL OR sm.world_valid_at = '')")
	}
	if sc, scArgs := scopeClauseFor(RecallInput{
		Scope:               in.Scope,
		SessionID:           in.SessionID,
		ProjectPath:         in.ProjectPath,
		IncludeRepoSiblings: in.IncludeRepoSiblings,
	}, "sm."); sc != "" {
		where = append(where, strings.TrimPrefix(sc, andFragment))
		args = append(args, scArgs...)
	}
	like := "%" + escapeLike(in.Query) + "%"
	cols := []string{"sm.subject", "sm.predicate", "sm.object", "sm.rule", "sm.title", "sm.target", "sm.reason"}
	content := make([]string, 0, len(cols))
	for _, col := range cols {
		content = append(content, col+` LIKE ? ESCAPE '\'`)
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
		e, err := scanEntry(rows)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entryToHistoryEntry(e))
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return &HistoryResult{Entries: entries}, nil
}
