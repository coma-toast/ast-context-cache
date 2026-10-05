package memory

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	selectActiveEntriesQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE 1=1`
	searchEntriesLikeQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR rule LIKE ?)`
	searchEntriesFTSQuery = `SELECT sm.ref, sm.kind, sm.scope, sm.session_id, sm.project_path, sm.subject, sm.predicate, sm.object, sm.rule,
		sm.valid_from, sm.valid_until, sm.superseded_by, sm.source_ref, sm.token_est, sm.access_count, sm.last_accessed_at, sm.created_at
		FROM structured_memory_fts f
		JOIN structured_memory sm ON sm.ref = f.ref
		WHERE structured_memory_fts MATCH ?`
	selectEntriesByRefsQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE ref IN (`
	selectActiveEntryTokensQuery    = `SELECT ref, token_est FROM structured_memory WHERE valid_until IS NULL OR valid_until = ''`
	invalidateEntryQuery            = `UPDATE structured_memory SET valid_until = datetime('now') WHERE ref = ?`
	invalidateActiveEntryQuery      = `UPDATE structured_memory SET valid_until = datetime('now') WHERE ref = ? AND (valid_until IS NULL OR valid_until = '')`
	updateEntryAccessQuery          = `UPDATE structured_memory SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE ref = ?`
	insertMemoryAccessQuery         = `INSERT INTO memory_access (ref, session_id, project_path, tool_name, tokens_returned) VALUES (?, ?, ?, ?, ?)`
	validAsOfClause                 = ` AND valid_from <= ? AND (valid_until IS NULL OR valid_until = '' OR valid_until > ?)`
	validNowClause                  = ` AND (valid_until IS NULL OR valid_until = '')`
	smValidAsOfClause               = ` AND sm.valid_from <= ? AND (sm.valid_until IS NULL OR sm.valid_until = '' OR sm.valid_until > ?)`
	smValidNowClause                = ` AND (sm.valid_until IS NULL OR sm.valid_until = '')`
	orderByAccessCreatedLimitClause = ` ORDER BY access_count DESC, created_at DESC LIMIT ?`
	orderByAccessLimitClause        = ` ORDER BY access_count DESC LIMIT ?`
	orderBySMAccessLimitClause      = ` ORDER BY sm.access_count DESC LIMIT ?`
	// Fragments that scopeClauseFor and projectMatch join around caller-chosen column names.
	andFragment          = ` AND `
	orFragment           = ` OR `
	eqParamFragment      = ` = ?`
	inParamsFragment     = ` IN (`
	scopeSessionFragment = ` = 'session' AND `
	scopeProjectFragment = ` = 'project'`
	scopeGlobalFragment  = ` = 'global'`
)

// RecallInput configures structured memory retrieval.
type RecallInput struct {
	Query          string
	SessionID      string
	ProjectPath    string
	Kinds          []Kind
	Scope          Scope  // optional filter
	AsOf           string // RFC3339 or SQLite datetime; empty = now (current facts only)
	Limit          int
	TokenBudget    int
	IncludeHistory bool // include superseded facts when as_of set
	// IncludeRepoSiblings widens project-scoped lookups to every indexed checkout of
	// the same repo, so a note taken in one worktree is recallable from a sibling
	// worktree of that repo sitting on a different branch.
	IncludeRepoSiblings bool
}

// RecallResult is token-efficient structured memory for agents.
type RecallResult struct {
	Entries        []Entry       `json:"entries"`
	Lines          []CompactLine `json:"lines"`
	Formatted      string        `json:"formatted"`
	TokensUsed     int           `json:"tokens_used"`
	TokensSavedEst int           `json:"tokens_saved_est"`
	RefsAccessed   int           `json:"refs_accessed"`
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
			return validAsOfClause, []any{asOf, asOf}
		}
		return validAsOfClause, []any{asOf, asOf}
	}
	return validNowClause, nil
}

// recallProjectPaths returns the project_path values a recall should match.
func recallProjectPaths(in RecallInput) []string {
	if in.ProjectPath == "" {
		return nil
	}
	if !in.IncludeRepoSiblings {
		return []string{in.ProjectPath}
	}
	paths := projectlinks.ResolveScopeWithRepoSiblings(in.ProjectPath, true)
	if len(paths) == 0 {
		return []string{in.ProjectPath}
	}
	return paths
}

// projectMatch builds a "<col> = ?" or "<col> IN (?,?)" fragment for paths.
func projectMatch(col string, paths []string) (string, []any) {
	switch len(paths) {
	case 0:
		return "", nil
	case 1:
		return col + eqParamFragment, []any{paths[0]}
	}
	ph := strings.TrimSuffix(strings.Repeat("?,", len(paths)), ",")
	args := make([]any, len(paths))
	for i, p := range paths {
		args[i] = p
	}
	return col + inParamsFragment + ph + ")", args
}

func scopeClause(in RecallInput) (string, []any) {
	return scopeClauseFor(in, "")
}

// scopeClauseFor builds the scope filter; prefix qualifies columns for joins.
func scopeClauseFor(in RecallInput, prefix string) (string, []any) {
	projectCol := prefix + "project_path"
	scopeCol := prefix + "scope"
	sessionCol := prefix + "session_id"
	projectFrag, projectArgs := projectMatch(projectCol, recallProjectPaths(in))

	if in.Scope != "" {
		switch in.Scope {
		case ScopeSession:
			return andFragment + scopeCol + scopeSessionFragment + sessionCol + eqParamFragment, []any{in.SessionID}
		case ScopeProject:
			if projectFrag == "" {
				return andFragment + scopeCol + scopeProjectFragment, nil
			}
			return andFragment + scopeCol + scopeProjectFragment + andFragment + projectFrag, projectArgs
		case ScopeGlobal:
			return andFragment + scopeCol + scopeGlobalFragment, nil
		}
	}
	var parts []string
	var args []any
	parts = append(parts, scopeCol+scopeGlobalFragment)
	if in.SessionID != "" {
		parts = append(parts, "("+scopeCol+scopeSessionFragment+sessionCol+eqParamFragment+")")
		args = append(args, in.SessionID)
	}
	if projectFrag != "" {
		parts = append(parts, "("+scopeCol+scopeProjectFragment+andFragment+projectFrag+")")
		args = append(args, projectArgs...)
	}
	if len(parts) == 1 && in.SessionID == "" && in.ProjectPath == "" {
		return "", nil
	}
	return andFragment + "(" + strings.Join(parts, orFragment) + ")", args
}

func listActiveEntries(in RecallInput) ([]Entry, error) {
	q := selectActiveEntriesQuery
	var args []any
	if clause, a := validityClause(in.AsOf, in.IncludeHistory); clause != "" {
		q += clause
		args = append(args, a...)
	}
	if clause, a := scopeClause(in); clause != "" {
		q += clause
		args = append(args, a...)
	}
	q += orderByAccessCreatedLimitClause
	args = append(args, in.Limit*2)
	return queryEntries(q, args...)
}

func searchEntries(in RecallInput, emb embedder.Interface) ([]Entry, error) {
	fts, _ := searchFTS(in)
	if len(fts) > 0 {
		return fts, nil
	}
	likeQ := `%` + in.Query + `%`
	q := searchEntriesLikeQuery
	args := []any{likeQ, likeQ, likeQ, likeQ}
	if clause, a := validityClause(in.AsOf, in.IncludeHistory); clause != "" {
		q += clause
		args = append(args, a...)
	}
	if clause, a := scopeClause(in); clause != "" {
		q += clause
		args = append(args, a...)
	}
	q += orderByAccessLimitClause
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
	q := searchEntriesFTSQuery
	args := []any{ftsQuery}
	if in.AsOf != "" {
		q += smValidAsOfClause
		args = append(args, in.AsOf, in.AsOf)
	} else {
		q += smValidNowClause
	}
	if clause, a := scopeClauseFor(in, "sm."); clause != "" {
		q += clause
		args = append(args, a...)
	}
	q += orderBySMAccessLimitClause
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
	q := selectEntriesByRefsQuery + placeholders + ")"
	args := make([]any, len(refs))
	for i, r := range refs {
		args[i] = r
	}
	return queryEntries(q, args...)
}

func queryEntries(q string, args ...any) ([]Entry, error) {
	rows, err := db.ContextDB.Query(q, args...)
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

// ForgetResult reports invalidated entries. For refs mode the per-ref lists
// say what happened to every ref passed in, so nothing is dropped silently.
type ForgetResult struct {
	InvalidatedRefs    int      `json:"invalidated_refs"`
	VirtualTokensFreed int      `json:"virtual_tokens_freed"`
	Invalidated        []string `json:"invalidated,omitempty"`
	NotFound           []string `json:"not_found,omitempty"`
	AlreadyInvalid     []string `json:"already_invalid,omitempty"`
	ScopeMismatch      []string `json:"scope_mismatch,omitempty"`
}

// Forget soft-invalidates structured memory.
func Forget(in ForgetInput) (*ForgetResult, error) {
	if in.All && len(in.Refs) > 0 {
		return nil, errs.NewCode(errs.CodeInvalidInput, "pass either refs or all=true, not both")
	}
	switch in.Scope {
	case "", ScopeSession, ScopeProject, ScopeGlobal:
	default:
		return nil, errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("invalid scope: %s (want session, project, or global)", in.Scope), "scope", in.Scope)
	}
	if in.All {
		rows, err := db.ContextDB.Query(selectActiveEntryTokensQuery)
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
		logger.Info("Invalidated all structured memory across every session and project", "invalidated", len(refs), "tokens", tokens)
		return &ForgetResult{InvalidatedRefs: len(refs), VirtualTokensFreed: tokens}, nil
	}
	if len(in.Refs) > 0 {
		return forgetRefs(in)
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
	return nil, errs.NewCode(errs.CodeInvalidInput, "refs, subject, or all=true required")
}

func invalidateRef(ref string) {
	db.ContextDB.Exec(invalidateEntryQuery, ref)
}

// forgetRefs invalidates exactly the named refs. Each ref's scope comes from its
// stored row, so callers need not pass scope/session_id; when they do pass a
// scope it acts as a guard (a ref outside it is reported, not invalidated).
// Unknown and already-invalid refs are reported back rather than counted as 0.
func forgetRefs(in ForgetInput) (*ForgetResult, error) {
	res := &ForgetResult{}
	seen := map[string]bool{}
	for _, ref := range in.Refs {
		ref = strings.TrimSpace(ref)
		if ref == "" || seen[ref] {
			continue
		}
		seen[ref] = true
		e, err := entryByRef(ref)
		if errors.Is(err, sql.ErrNoRows) {
			res.NotFound = append(res.NotFound, ref)
			continue
		}
		if err != nil {
			return nil, errs.WrapMessage("failed to look up memory", err, "ref", ref)
		}
		if e.ValidUntil != "" {
			res.AlreadyInvalid = append(res.AlreadyInvalid, ref)
			continue
		}
		if !refInScope(e, in) {
			res.ScopeMismatch = append(res.ScopeMismatch, ref)
			continue
		}
		if _, err := db.ContextDB.Exec(invalidateActiveEntryQuery, ref); err != nil {
			return nil, errs.WrapMessage("failed to invalidate memory", err, "ref", ref)
		}
		res.Invalidated = append(res.Invalidated, ref)
		res.InvalidatedRefs++
		res.VirtualTokensFreed += e.TokenEst
	}
	return res, nil
}

// refInScope reports whether e passes the caller's optional scope guard. With no
// scope given every ref passes; with scope=session and a session_id, the ref
// must belong to that session.
func refInScope(e Entry, in ForgetInput) bool {
	if in.Scope == "" {
		return true
	}
	if e.Scope != in.Scope {
		return false
	}
	if in.Scope == ScopeSession && in.SessionID != "" && e.SessionID != in.SessionID {
		return false
	}
	return true
}

// RecordAccess tracks recall for dashboard stats.
func RecordAccess(ref, sessionID, projectPath, tool string, tokens int) {
	db.ContextDB.Exec(updateEntryAccessQuery, ref)
	db.DB.Exec(insertMemoryAccessQuery,
		ref, nullIfEmpty(sessionID), nullIfEmpty(projectPath), tool, tokens)
}
