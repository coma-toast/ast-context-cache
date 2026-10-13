package memory

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// v2Enabled reports whether the feature_memory_v2 flag gates MEM-6..MEM-10
// (recency decay ranking, TTL expiry, debug demotion output, compact lines
// default with hard caps). History, restore and world time are always on.
func v2Enabled() bool { return flags.Enabled(flags.KeyMemoryV2) }

const (
	selectActiveEntriesQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at,
		invalidated_reason, world_valid_at, world_invalid_at, expires_at, quarantined
		FROM structured_memory WHERE 1=1`
	selectEntriesByRefsQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at,
		invalidated_reason, world_valid_at, world_invalid_at, expires_at, quarantined
		FROM structured_memory WHERE ref IN (`
	selectActiveEntryTokensQuery = `SELECT ref, token_est FROM structured_memory WHERE (valid_until IS NULL OR valid_until = '')`
	invalidateEntryQuery         = `UPDATE structured_memory SET valid_until = datetime('now'), invalidated_reason = ? WHERE ref = ?`
	invalidateActiveEntryQuery   = `UPDATE structured_memory SET valid_until = datetime('now'), invalidated_reason = ? WHERE ref = ? AND (valid_until IS NULL OR valid_until = '')`
	updateEntryAccessQuery       = `UPDATE structured_memory SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE ref = ?`
	insertMemoryAccessQuery      = `INSERT INTO memory_access (ref, session_id, project_path, tool_name, tokens_returned) VALUES (?, ?, ?, ?, ?)`
	validAsOfClause              = ` AND valid_from <= ? AND (valid_until IS NULL OR valid_until = '' OR valid_until > ?)`
	validFromAsOfClause          = ` AND valid_from <= ?`
	validNowClause               = ` AND (valid_until IS NULL OR valid_until = '')`
	// MEM-5: world-time as_of clauses use COALESCE(world_valid_at, valid_from) so that
	// as_of queries use world time when present and fall back to system time when the
	// world columns are empty. The "now" clause stays on system time (valid_until).
	worldValidAsOfClause     = ` AND COALESCE(world_valid_at, valid_from) <= ? AND (COALESCE(world_invalid_at, valid_until) IS NULL OR COALESCE(world_invalid_at, valid_until) = '' OR COALESCE(world_invalid_at, valid_until) > ?)`
	worldValidFromAsOfClause = ` AND COALESCE(world_valid_at, valid_from) <= ?`
	smValidAsOfClause        = ` AND sm.valid_from <= ? AND (sm.valid_until IS NULL OR sm.valid_until = '' OR sm.valid_until > ?)`
	smValidFromAsOfClause    = ` AND sm.valid_from <= ?`
	smValidNowClause         = ` AND (sm.valid_until IS NULL OR sm.valid_until = '')`
	// MEM-5: world-time as_of clauses qualified for the structured_memory join.
	smWorldValidAsOfClause          = ` AND COALESCE(sm.world_valid_at, sm.valid_from) <= ? AND (COALESCE(sm.world_invalid_at, sm.valid_until) IS NULL OR COALESCE(sm.world_invalid_at, sm.valid_until) = '' OR COALESCE(sm.world_invalid_at, sm.valid_until) > ?)`
	smWorldValidFromAsOfClause      = ` AND COALESCE(sm.world_valid_at, sm.valid_from) <= ?`
	orderByAccessCreatedLimitClause = ` ORDER BY access_count DESC, created_at DESC, ref LIMIT ?`
	andKindInClause                 = ` AND kind IN (`
	andSMKindInClause               = ` AND sm.kind IN (`
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
	Query       string
	SessionID   string
	ProjectPath string
	Kinds       []Kind
	Scope       Scope  // optional filter
	AsOf        string // RFC3339 or SQLite datetime; empty = now (current facts only)
	// AsOfSystem uses system valid_from/valid_until for the as_of query instead of
	// world time. When false (default), as_of uses world_valid_at/world_invalid_at
	// when present, falling back to system time.
	AsOfSystem  bool
	Limit       int
	TokenBudget int
	// IncludeHistory drops the valid_until (superseded/forgotten) filter. With
	// AsOf set it still excludes entries that only became valid after AsOf.
	IncludeHistory bool
	// IncludeRepoSiblings widens project-scoped lookups to every indexed checkout of
	// the same repo, so a note taken in one worktree is recallable from a sibling
	// worktree of that repo sitting on a different branch.
	IncludeRepoSiblings bool
	// Debug lists entries that matched but were demoted below the budget, with their
	// scores and the reason (MEM-9).
	Debug bool
	// Expand returns full entry bodies instead of compact one-line summaries (MEM-10).
	Expand bool
}

// DemotedEntry is a matched entry that did not make the final output, with the reason.
type DemotedEntry struct {
	Ref    string  `json:"ref"`
	Line   string  `json:"line"`
	Score  float64 `json:"score"`
	Reason string  `json:"reason"` // "budget" | "limit" | "decay"
}

// RecallResult is token-efficient structured memory for agents.
type RecallResult struct {
	Entries        []Entry        `json:"entries,omitempty"`
	Lines          []CompactLine  `json:"lines,omitempty"`
	Formatted      string         `json:"formatted,omitempty"`
	Demoted        []DemotedEntry `json:"demoted,omitempty"`
	TokensUsed     int            `json:"tokens_used"`
	TokensSavedEst int            `json:"tokens_saved_est"`
	RefsAccessed   int            `json:"refs_accessed"`
}

// validityClauses holds the validity filters for one column qualification, so
// joined and unjoined queries share validityClause.
type validityClauses struct{ now, asOf, fromAsOf, worldAsOf, worldFromAsOf string }

var (
	entryValidity   = validityClauses{now: validNowClause, asOf: validAsOfClause, fromAsOf: validFromAsOfClause, worldAsOf: worldValidAsOfClause, worldFromAsOf: worldValidFromAsOfClause}
	smEntryValidity = validityClauses{now: smValidNowClause, asOf: smValidAsOfClause, fromAsOf: smValidFromAsOfClause, worldAsOf: smWorldValidAsOfClause, worldFromAsOf: smWorldValidFromAsOfClause}
)

// Recall returns compact valid facts and procedures matching query within token budget.
func Recall(in RecallInput, emb embedder.Interface) (*RecallResult, error) {
	in.Query = strings.TrimSpace(in.Query)
	if in.Limit <= 0 {
		in.Limit = 10
	}
	if in.Limit > 50 {
		in.Limit = 50 // hard cap: 50 lines
	}
	if in.TokenBudget <= 0 {
		in.TokenBudget = 800
	}
	if in.TokenBudget > 2000 && v2Enabled() {
		in.TokenBudget = 2000 // hard cap: 2,000 tokens
	}
	var entries []Entry
	var scores map[string]float64
	var err error
	if in.Query != "" {
		entries, scores, err = searchEntries(in, emb)
	} else {
		entries, err = listActiveEntries(in)
	}
	if err != nil {
		return nil, err
	}
	v2 := v2Enabled()
	var demotedLimit []DemotedEntry
	if len(entries) > in.Limit {
		if v2 && in.Debug {
			for _, e := range entries[in.Limit:] {
				demotedLimit = append(demotedLimit, DemotedEntry{
					Ref:    e.Ref,
					Line:   FormatLine(e),
					Score:  scores[e.Ref],
					Reason: "limit",
				})
			}
		}
		entries = entries[:in.Limit]
	}
	budgeted, tokensUsed, savedEst, demotedBudget := applyTokenBudget(entries, in.TokenBudget, v2 && in.Debug)
	lines := make([]CompactLine, 0, len(budgeted))
	for _, e := range budgeted {
		line := FormatLine(e)
		if line == "" {
			continue
		}
		RecordAccess(e.Ref, in.SessionID, in.ProjectPath, "recall_memory", estimateEntryTokens(e))
		lines = append(lines, CompactLine{Ref: e.Ref, Kind: e.Kind, Line: line, Score: scores[e.Ref]})
	}
	var demoted []DemotedEntry
	if v2 {
		demoted = append(demotedBudget, demotedLimit...)
	}

	// MEM-10: expand=true adds formatted string; default (expand=false) drops formatted.
	var formatted string
	if v2 && in.Expand {
		formatted = formatLines(budgeted)
	}

	return &RecallResult{
		Entries:        budgeted,
		Lines:          lines,
		Formatted:      formatted,
		Demoted:        demoted,
		TokensUsed:     tokensUsed,
		TokensSavedEst: savedEst,
		RefsAccessed:   len(lines),
	}, nil
}

// applyTokenBudget applies the token budget cap, returning the entries that fit within
// the budget, the number of tokens used, the estimated tokens saved by truncation, and
// the entries that were demoted below the budget (MEM-9).
func applyTokenBudget(entries []Entry, budget int, debug bool) ([]Entry, int, int, []DemotedEntry) {
	var out []Entry
	used := 0
	saved := 0
	var demoted []DemotedEntry
	for _, e := range entries {
		tok := estimateEntryTokens(e)
		fullEst := tok * 8
		if used+tok > budget && len(out) > 0 {
			if debug {
				demoted = append(demoted, DemotedEntry{
					Ref:    e.Ref,
					Line:   FormatLine(e),
					Score:  0,
					Reason: "budget",
				})
			}
			break
		}
		out = append(out, e)
		used += tok
		saved += fullEst - tok
	}
	if saved < 0 {
		saved = 0
	}
	return out, used, saved, demoted
}

// validityClause builds the validity filter from c's column qualification.
func validityClause(in RecallInput, c validityClauses) (string, []any) {
	switch {
	case in.AsOf != "" && in.IncludeHistory:
		if in.AsOfSystem {
			return c.fromAsOf, []any{in.AsOf}
		}
		return c.worldFromAsOf, []any{in.AsOf}
	case in.AsOf != "":
		if in.AsOfSystem {
			return c.asOf, []any{in.AsOf, in.AsOf}
		}
		return c.worldAsOf, []any{in.AsOf, in.AsOf}
	case in.IncludeHistory:
		return "", nil
	}
	return c.now, nil
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

// kindClause builds the kind filter; prefix qualifies the column for joins.
func kindClause(kinds []Kind, prefix string) (string, []any) {
	if len(kinds) == 0 {
		return "", nil
	}
	clause := andKindInClause
	if prefix != "" {
		clause = andSMKindInClause
	}
	args := make([]any, len(kinds))
	for i, k := range kinds {
		args[i] = string(k)
	}
	return clause + strings.TrimSuffix(strings.Repeat("?,", len(kinds)), ",") + ")", args
}

// filterClauses joins the validity, scope and kind filters every recall query applies;
// prefix qualifies columns for joins and v must match it.
func filterClauses(in RecallInput, prefix string, v validityClauses) (string, []any) {
	validity, args := validityClause(in, v)
	scope, scopeArgs := scopeClauseFor(in, prefix)
	kind, kindArgs := kindClause(in.Kinds, prefix)
	args = append(append(args, scopeArgs...), kindArgs...)
	return validity + scope + kind, args
}

func listActiveEntries(in RecallInput) ([]Entry, error) {
	clause, args := filterClauses(in, "", entryValidity)
	return queryEntries(selectActiveEntriesQuery+clause+orderByAccessCreatedLimitClause, append(args, in.Limit*2)...)
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
	// Confirm allows all=true with no scope, session or project, which invalidates every
	// active entry across all sessions and projects.
	Confirm bool
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
		return forgetAll(in)
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

// forgetAll invalidates every active entry in the given scope, session or project. With none
// of them it needs Confirm, since it then wipes memory across every session and project.
func forgetAll(in ForgetInput) (*ForgetResult, error) {
	clause, args := forgetScopeClause(in)
	if clause == "" && !in.Confirm {
		return nil, errs.NewCode(errs.CodeInvalidInput, "all=true without scope requires confirm=true")
	}
	rows, err := db.ContextDB.Query(selectActiveEntryTokensQuery+clause, args...)
	if err != nil {
		return nil, errs.WrapMessage("failed to select memory to forget", err)
	}
	var refs []string
	tokens := 0
	for rows.Next() {
		var ref string
		var tok int
		if rows.Scan(&ref, &tok) == nil {
			refs = append(refs, ref)
			tokens += tok
		}
	}
	rows.Close()
	for _, ref := range refs {
		invalidateRef(ref, ReasonUserForget)
	}
	logger.Info("Invalidated all structured memory in scope", "invalidated", len(refs), "tokens", tokens,
		"scope", in.Scope, "session_id", in.SessionID, "project_path", in.ProjectPath)
	return &ForgetResult{InvalidatedRefs: len(refs), VirtualTokensFreed: tokens}, nil
}

// forgetScopeClause limits forget all=true. An explicit scope uses the recall scope filter;
// otherwise it matches the given session's and project's entries but never global ones.
func forgetScopeClause(in ForgetInput) (string, []any) {
	rin := RecallInput{Scope: in.Scope, SessionID: in.SessionID, ProjectPath: in.ProjectPath}
	if in.Scope != "" {
		return scopeClause(rin)
	}
	var parts []string
	var args []any
	if in.SessionID != "" {
		parts = append(parts, "(scope"+scopeSessionFragment+"session_id"+eqParamFragment+")")
		args = append(args, in.SessionID)
	}
	if in.ProjectPath != "" {
		parts = append(parts, "(scope"+scopeProjectFragment+andFragment+"project_path"+eqParamFragment+")")
		args = append(args, in.ProjectPath)
	}
	if len(parts) == 0 {
		return "", nil
	}
	return andFragment + "(" + strings.Join(parts, orFragment) + ")", args
}

func invalidateRef(ref, reason string) {
	invalidate(db.ContextDB, ref, reason, "")
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
		if err := invalidate(db.ContextDB, ref, ReasonUserForget, ""); err != nil {
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
