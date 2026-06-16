package memory

import (
	"errors"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// RecallInput configures structured memory retrieval.
type RecallInput struct {
	Query        string
	SessionID    string
	ProjectPath  string
	Kinds        []Kind
	Scope        Scope // optional filter
	AsOf         string // RFC3339 or SQLite datetime; empty = now (current facts only)
	Limit        int
	TokenBudget  int
	IncludeHistory bool // include superseded facts when as_of set
}

// RecallResult is token-efficient structured memory for agents.
type RecallResult struct {
	Entries         []Entry        `json:"entries"`
	Lines           []CompactLine  `json:"lines"`
	Formatted       string         `json:"formatted"`
	TokensUsed      int            `json:"tokens_used"`
	TokensSavedEst  int            `json:"tokens_saved_est"`
	RefsAccessed    int            `json:"refs_accessed"`
}

// Recall returns compact valid facts and procedures matching query within token budget.
func Recall(in RecallInput, emb embedder.Interface) (*RecallResult, error) {
	in.Query = strings.TrimSpace(in.Query)
	if in.Limit <= 0 {
		in.Limit = 10
	}
	if in.TokenBudget <= 0 {
		in.TokenBudget = 800
	}
	var entries []Entry
	var err error
	if in.Query != "" {
		entries, err = searchEntries(in, emb)
	} else {
		entries, err = listActiveEntries(in)
	}
	if err != nil {
		return nil, err
	}
	entries = filterKinds(entries, in.Kinds)
	budgeted, tokensUsed, savedEst := applyTokenBudget(entries, in.TokenBudget)
	lines := make([]CompactLine, 0, len(budgeted))
	for _, e := range budgeted {
		line := FormatLine(e)
		if line == "" {
			continue
		}
		RecordAccess(e.Ref, in.SessionID, in.ProjectPath, "recall_memory", estimateEntryTokens(e))
		lines = append(lines, CompactLine{Ref: e.Ref, Kind: e.Kind, Line: line})
	}
	return &RecallResult{
		Entries:        budgeted,
		Lines:          lines,
		Formatted:      formatLines(budgeted),
		TokensUsed:     tokensUsed,
		TokensSavedEst: savedEst,
		RefsAccessed:   len(lines),
	}, nil
}

func filterKinds(entries []Entry, kinds []Kind) []Entry {
	if len(kinds) == 0 {
		return entries
	}
	want := map[Kind]bool{}
	for _, k := range kinds {
		want[k] = true
	}
	out := make([]Entry, 0, len(entries))
	for _, e := range entries {
		if want[e.Kind] {
			out = append(out, e)
		}
	}
	return out
}

func applyTokenBudget(entries []Entry, budget int) ([]Entry, int, int) {
	var out []Entry
	used := 0
	saved := 0
	for _, e := range entries {
		tok := estimateEntryTokens(e)
		fullEst := tok * 8
		if used+tok > budget && len(out) > 0 {
			break
		}
		out = append(out, e)
		used += tok
		saved += fullEst - tok
	}
	if saved < 0 {
		saved = 0
	}
	return out, used, saved
}

func validityClause(asOf string, includeHistory bool) (string, []any) {
	if asOf != "" {
		if includeHistory {
			return ` AND valid_from <= ? AND (valid_until IS NULL OR valid_until = '' OR valid_until > ?)`, []any{asOf, asOf}
		}
		return ` AND valid_from <= ? AND (valid_until IS NULL OR valid_until = '' OR valid_until > ?)`, []any{asOf, asOf}
	}
	return ` AND (valid_until IS NULL OR valid_until = '')`, nil
}

func scopeClause(sessionID, projectPath string, scope Scope) (string, []any) {
	if scope != "" {
		switch scope {
		case ScopeSession:
			return ` AND scope = 'session' AND session_id = ?`, []any{sessionID}
		case ScopeProject:
			return ` AND scope = 'project' AND project_path = ?`, []any{projectPath}
		case ScopeGlobal:
			return ` AND scope = 'global'`, nil
		}
	}
	var parts []string
	var args []any
	parts = append(parts, `scope = 'global'`)
	if sessionID != "" {
		parts = append(parts, `(scope = 'session' AND session_id = ?)`)
		args = append(args, sessionID)
	}
	if projectPath != "" {
		parts = append(parts, `(scope = 'project' AND project_path = ?)`)
		args = append(args, projectPath)
	}
	if len(parts) == 1 && sessionID == "" && projectPath == "" {
		return "", nil
	}
	return ` AND (` + strings.Join(parts, ` OR `) + `)`, args
}

func listActiveEntries(in RecallInput) ([]Entry, error) {
	q := `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE 1=1`
	var args []any
	if clause, a := validityClause(in.AsOf, in.IncludeHistory); clause != "" {
		q += clause
		args = append(args, a...)
	}
	if clause, a := scopeClause(in.SessionID, in.ProjectPath, in.Scope); clause != "" {
		q += clause
		args = append(args, a...)
	}
	q += ` ORDER BY access_count DESC, created_at DESC LIMIT ?`
	args = append(args, in.Limit*2)
	return queryEntries(q, args...)
}

func searchEntries(in RecallInput, emb embedder.Interface) ([]Entry, error) {
	fts, _ := searchFTS(in)
	if len(fts) > 0 {
		return fts, nil
	}
	likeQ := `%` + in.Query + `%`
	q := `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR rule LIKE ?)`
	args := []any{likeQ, likeQ, likeQ, likeQ}
	if clause, a := validityClause(in.AsOf, in.IncludeHistory); clause != "" {
		q += clause
		args = append(args, a...)
	}
	if clause, a := scopeClause(in.SessionID, in.ProjectPath, in.Scope); clause != "" {
		q += clause
		args = append(args, a...)
	}
	q += ` ORDER BY access_count DESC LIMIT ?`
	args = append(args, in.Limit*2)
	entries, err := queryEntries(q, args...)
	if err != nil || len(entries) > 0 || emb == nil {
		return entries, err
	}
	return vectorSearch(in, emb)
}

func searchFTS(in RecallInput) ([]Entry, error) {
	ftsQuery := search.BuildFTSQuery(search.QueryTerms(in.Query))
	if ftsQuery == "" {
		return nil, nil
	}
	q := `SELECT sm.ref, sm.kind, sm.scope, sm.session_id, sm.project_path, sm.subject, sm.predicate, sm.object, sm.rule,
		sm.valid_from, sm.valid_until, sm.superseded_by, sm.source_ref, sm.token_est, sm.access_count, sm.last_accessed_at, sm.created_at
		FROM structured_memory_fts f
		JOIN structured_memory sm ON sm.ref = f.ref
		WHERE structured_memory_fts MATCH ?`
	args := []any{ftsQuery}
	if in.AsOf != "" {
		q += ` AND sm.valid_from <= ? AND (sm.valid_until IS NULL OR sm.valid_until = '' OR sm.valid_until > ?)`
		args = append(args, in.AsOf, in.AsOf)
	} else {
		q += ` AND (sm.valid_until IS NULL OR sm.valid_until = '')`
	}
	if in.Scope != "" {
		switch in.Scope {
		case ScopeSession:
			q += ` AND sm.scope = 'session' AND sm.session_id = ?`
			args = append(args, in.SessionID)
		case ScopeProject:
			q += ` AND sm.scope = 'project' AND sm.project_path = ?`
			args = append(args, in.ProjectPath)
		case ScopeGlobal:
			q += ` AND sm.scope = 'global'`
		}
	} else {
		var parts []string
		parts = append(parts, `sm.scope = 'global'`)
		if in.SessionID != "" {
			parts = append(parts, `(sm.scope = 'session' AND sm.session_id = ?)`)
			args = append(args, in.SessionID)
		}
		if in.ProjectPath != "" {
			parts = append(parts, `(sm.scope = 'project' AND sm.project_path = ?)`)
			args = append(args, in.ProjectPath)
		}
		if len(parts) > 1 || in.SessionID != "" || in.ProjectPath != "" {
			q += ` AND (` + strings.Join(parts, ` OR `) + `)`
		}
	}
	q += ` ORDER BY sm.access_count DESC LIMIT ?`
	args = append(args, in.Limit*2)
	return queryEntries(q, args...)
}

func vectorSearch(in RecallInput, emb embedder.Interface) ([]Entry, error) {
	vec, err := emb.EmbedSingle(in.Query)
	if err != nil {
		return nil, err
	}
	scored := search.Cache.SearchMemory(vec, in.SessionID, in.Limit*2)
	var refs []string
	for _, s := range scored {
		if ref, _ := s.Data["ref"].(string); ref != "" {
			refs = append(refs, ref)
		}
	}
	if len(refs) == 0 {
		return nil, nil
	}
	placeholders := strings.Repeat("?,", len(refs))
	placeholders = placeholders[:len(placeholders)-1]
	q := `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE ref IN (` + placeholders + `)`
	args := make([]any, len(refs))
	for i, r := range refs {
		args[i] = r
	}
	return queryEntries(q, args...)
}

func queryEntries(q string, args ...any) ([]Entry, error) {
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Entry
	for rows.Next() {
		e, err := scanEntry(rows)
		if err != nil {
			continue
		}
		out = append(out, e)
	}
	return out, nil
}

// ForgetInput invalidates memory entries (soft delete via valid_until).
type ForgetInput struct {
	Refs        []string
	SessionID   string
	ProjectPath string
	Subject     string
	Predicate   string
	Scope       Scope
	All         bool
}

// ForgetResult reports invalidated entries.
type ForgetResult struct {
	InvalidatedRefs      int `json:"invalidated_refs"`
	VirtualTokensFreed   int `json:"virtual_tokens_freed"`
}

// Forget soft-invalidates structured memory.
func Forget(in ForgetInput) (*ForgetResult, error) {
	if in.All {
		rows, err := db.DB.Query(`SELECT ref, token_est FROM structured_memory WHERE valid_until IS NULL OR valid_until = ''`)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		var refs []string
		tokens := 0
		for rows.Next() {
			var ref string
			var tok int
			rows.Scan(&ref, &tok)
			refs = append(refs, ref)
			tokens += tok
		}
		for _, ref := range refs {
			invalidateRef(ref)
		}
		return &ForgetResult{InvalidatedRefs: len(refs), VirtualTokensFreed: tokens}, nil
	}
	if len(in.Refs) > 0 {
		tokens := 0
		count := 0
		for _, ref := range in.Refs {
			var tok int
			if db.DB.QueryRow(`SELECT token_est FROM structured_memory WHERE ref = ?`, ref).Scan(&tok) == nil {
				invalidateRef(ref)
				tokens += tok
				count++
			}
		}
		return &ForgetResult{InvalidatedRefs: count, VirtualTokensFreed: tokens}, nil
	}
	if in.Subject != "" {
		pred := in.Predicate
		if pred == "" {
			pred = "is"
		}
		sc := in.Scope
		if sc == "" {
			sc = ScopeSession
		}
		refs, _ := invalidateConflicting(sc, in.SessionID, in.ProjectPath, in.Subject, pred, "")
		return &ForgetResult{InvalidatedRefs: len(refs)}, nil
	}
	return nil, errors.New("refs, subject, or all=true required")
}

func invalidateRef(ref string) {
	db.DB.Exec(`UPDATE structured_memory SET valid_until = datetime('now') WHERE ref = ?`, ref)
}

// RecordAccess tracks recall for dashboard stats.
func RecordAccess(ref, sessionID, projectPath, tool string, tokens int) {
	db.DB.Exec(`UPDATE structured_memory SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE ref = ?`, ref)
	db.DB.Exec(`INSERT INTO memory_access (ref, session_id, project_path, tool_name, tokens_returned) VALUES (?, ?, ?, ?, ?)`,
		ref, nullIfEmpty(sessionID), nullIfEmpty(projectPath), tool, tokens)
}
