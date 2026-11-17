package memory

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

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
	bodyTruncSuffix              = "\n... [truncated; request smaller budget or view in Dashboard]"
	selectActiveEntryTokensQuery = `SELECT ref, token_est FROM structured_memory WHERE (valid_until IS NULL OR valid_until = '')`
	invalidateEntryQuery         = `UPDATE structured_memory SET valid_until = datetime('now'), invalidated_reason = ? WHERE ref = ?`
	invalidateActiveEntryQuery   = `UPDATE structured_memory SET valid_until = datetime('now'), invalidated_reason = ? WHERE ref = ? AND (valid_until IS NULL OR valid_until = '')`
	updateEntryAccessQuery       = `UPDATE structured_memory SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE ref = ?`
	insertMemoryAccessQuery      = `INSERT INTO memory_access (ref, session_id, project_path, tool_name, tokens_returned) VALUES (?, ?, ?, ?, ?)`
	validAsOfClause              = ` AND valid_from <= ? AND (valid_until IS NULL OR valid_until = '' OR valid_until > ?)`
	validFromAsOfClause          = ` AND valid_from <= ?`
	validNowClause               = ` AND (valid_until IS NULL OR valid_until = '')`
	validNowOrSupersededClause   = ` AND ((valid_until IS NULL OR valid_until = '') OR invalidated_reason IN ('contradiction', 'superseded') OR (superseded_by IS NOT NULL AND superseded_by != ''))`
	// MEM-5: world-time as_of clauses use COALESCE(world_valid_at, valid_from) so that
	// as_of queries use world time when present and fall back to system time when the
	// world columns are empty. The "now" clause stays on system time (valid_until).
	worldValidAsOfClause     = ` AND COALESCE(world_valid_at, valid_from) <= ? AND (COALESCE(world_invalid_at, valid_until) IS NULL OR COALESCE(world_invalid_at, valid_until) = '' OR COALESCE(world_invalid_at, valid_until) > ?)`
	worldValidFromAsOfClause = ` AND COALESCE(world_valid_at, valid_from) <= ?`
	smValidAsOfClause        = ` AND sm.valid_from <= ? AND (sm.valid_until IS NULL OR sm.valid_until = '' OR sm.valid_until > ?)`
	smValidFromAsOfClause    = ` AND sm.valid_from <= ?`
	smValidNowClause         = ` AND (sm.valid_until IS NULL OR sm.valid_until = '')`
	smValidNowOrSuperseded   = ` AND ((sm.valid_until IS NULL OR sm.valid_until = '') OR sm.invalidated_reason IN ('contradiction', 'superseded') OR (sm.superseded_by IS NOT NULL AND sm.superseded_by != ''))`
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
	// KB filters (Tier-1/Tier-2 recall).
	Refs               []string
	Tags               []string
	Actor              string
	Since              string
	Until              string
	VerifiedOnly       bool
	IncludeQuarantined bool
	IncludeSuperseded  bool
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
type validityClauses struct{ now, nowOrSuperseded, asOf, fromAsOf, worldAsOf, worldFromAsOf string }

var (
	entryValidity   = validityClauses{now: validNowClause, nowOrSuperseded: validNowOrSupersededClause, asOf: validAsOfClause, fromAsOf: validFromAsOfClause, worldAsOf: worldValidAsOfClause, worldFromAsOf: worldValidFromAsOfClause}
	smEntryValidity = validityClauses{now: smValidNowClause, nowOrSuperseded: smValidNowOrSuperseded, asOf: smValidAsOfClause, fromAsOf: smValidFromAsOfClause, worldAsOf: smWorldValidAsOfClause, worldFromAsOf: smWorldValidFromAsOfClause}
)

// Recall returns compact valid facts and procedures matching query within token budget.
func Recall(in RecallInput, emb embedder.Interface) (*RecallResult, error) {
	in.Query = strings.TrimSpace(in.Query)
	in.Actor = strings.TrimSpace(in.Actor)
	if in.Since = strings.TrimSpace(in.Since); in.Since != "" {
		norm, err := db.NormalizeSQLTime(in.Since)
		if err != nil {
			return nil, err
		}
		in.Since = norm
	}
	if in.Until = strings.TrimSpace(in.Until); in.Until != "" {
		norm, err := db.NormalizeSQLTime(in.Until)
		if err != nil {
			return nil, err
		}
		in.Until = norm
	}

	if in.Limit <= 0 {
		if len(in.Refs) > 10 {
			in.Limit = len(in.Refs)
		} else {
			in.Limit = 10
		}
	}
	if in.Limit > 50 {
		in.Limit = 50 // hard cap: 50 lines
	}

	fullTier2 := in.Expand || len(in.Refs) > 0
	if in.TokenBudget <= 0 {
		in.TokenBudget = 800
	}
	maxBudget := 2000
	if fullTier2 {
		maxBudget = 8000
	}
	if in.TokenBudget > maxBudget && v2Enabled() {
		in.TokenBudget = maxBudget
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
	budgeted, tokensUsed, savedEst, demotedBudget := applyTokenBudget(entries, in.TokenBudget, v2 && in.Debug, fullTier2)
	lines := make([]CompactLine, 0, len(budgeted))
	for _, e := range budgeted {
		cl := FormatCompactLine(e, scores[e.Ref])
		if cl.Line == "" {
			continue
		}
		RecordAccess(e.Ref, in.SessionID, in.ProjectPath, "recall_memory", returnedTokens(e, fullTier2))
		lines = append(lines, cl)
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

func returnedTokens(e Entry, fullTier2 bool) int {
	if fullTier2 {
		return estimateEntryTokens(e)
	}
	tok := db.EstimateTokens(FormatLine(e))
	if tok <= 0 && e.TokenEst > 0 {
		return e.TokenEst
	}
	return tok
}

// truncateEntryToBudget trims e.Body at a UTF-8 rune boundary and appends
// bodyTruncSuffix so a single oversized entry fits within budget.
func truncateEntryToBudget(e Entry, budget int) (Entry, int) {
	if e.Body == "" {
		return e, estimateEntryTokens(e)
	}
	e.TokenEst = 0
	header := FormatLine(e)
	var cuts []int
	cuts = append(cuts, 0)
	for i, r := range e.Body {
		w := utf8.RuneLen(r)
		if w < 0 {
			w = 1
		}
		cuts = append(cuts, i+w)
	}

	buildText := func(prefix string) string {
		body := prefix + bodyTruncSuffix
		if header != "" {
			return header + "\n" + body
		}
		return body
	}

	lo, hi := 0, len(cuts)-1
	best := -1
	for lo <= hi {
		mid := (lo + hi) / 2
		tok := db.EstimateTokens(buildText(e.Body[:cuts[mid]]))
		if tok <= budget {
			best = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}

	if best < 0 {
		e.Body = strings.TrimPrefix(bodyTruncSuffix, "\n")
	} else {
		e.Body = e.Body[:cuts[best]] + bodyTruncSuffix
	}
	tok := estimateEntryTokens(e)
	e.TokenEst = tok
	return e, tok
}

// applyTokenBudget applies the token budget cap, returning the entries that fit within
// the budget, the number of tokens used, the estimated tokens saved by truncation, and
// the entries that were demoted below the budget (MEM-9).
func applyTokenBudget(entries []Entry, budget int, debug bool, fullTier2 bool) ([]Entry, int, int, []DemotedEntry) {
	var out []Entry
	used := 0
	saved := 0
	var demoted []DemotedEntry
	for _, e := range entries {
		tok := returnedTokens(e, fullTier2)
		fullEst := e.TokenEst
		if fullEst < tok*8 {
			fullEst = tok * 8
		}
		if len(out) == 0 && fullTier2 && tok > budget && e.Body != "" {
			e, tok = truncateEntryToBudget(e, budget)
			if fullEst < tok {
				fullEst = tok
			}
		}
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
	case in.IncludeSuperseded:
		return c.nowOrSuperseded, nil
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

func refsClause(refs []string, prefix string) (string, []any) {
	if len(refs) == 0 {
		return "", nil
	}
	var clean []string
	for _, r := range refs {
		if r = strings.TrimSpace(r); r != "" {
			clean = append(clean, r)
		}
	}
	if len(clean) == 0 {
		return "", nil
	}
	args := make([]any, len(clean))
	for i, r := range clean {
		args[i] = r
	}
	return andFragment + prefix + "ref IN (" + strings.TrimSuffix(strings.Repeat("?,", len(clean)), ",") + ")", args
}

func escapeLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

func tagsClause(tags []string, prefix string) (string, []any) {
	norm := normalizeTags(tags)
	if len(norm) == 0 {
		return "", nil
	}
	col := "lower(COALESCE(" + prefix + "tags, ''))"
	var parts []string
	var args []any
	for _, t := range norm {
		b, _ := json.Marshal(t)
		jsonPat := "%" + escapeLike(string(b)) + "%"
		csvPat := "%," + escapeLike(t) + ",%"
		parts = append(parts, "("+col+` LIKE ? ESCAPE '\' OR ',' || REPLACE(`+col+`, ' ', '') || ',' LIKE ? ESCAPE '\')`)
		args = append(args, jsonPat, csvPat)
	}
	return andFragment + "(" + strings.Join(parts, orFragment) + ")", args
}

func actorClause(actor, prefix string) (string, []any) {
	actor = strings.TrimSpace(actor)
	if actor == "" {
		return "", nil
	}
	return andFragment + prefix + "actor = ?", []any{actor}
}

func timeRangeClause(since, until, prefix string) (string, []any) {
	var clause string
	var args []any
	tsCol := "COALESCE(NULLIF(" + prefix + "occurred_at,''), " + prefix + "created_at)"
	if since != "" {
		norm := since
		if n, err := db.NormalizeSQLTime(since); err == nil {
			norm = n
		}
		clause += andFragment + tsCol + " >= ?"
		args = append(args, norm)
	}
	if until != "" {
		norm := until
		if n, err := db.NormalizeSQLTime(until); err == nil {
			norm = n
		}
		clause += andFragment + tsCol + " <= ?"
		args = append(args, norm)
	}
	return clause, args
}

func verifiedClause(verifiedOnly bool, prefix string) string {
	if !verifiedOnly {
		return ""
	}
	return andFragment + "COALESCE(" + prefix + "verified, 0) = 1"
}

func quarantineClause(includeQuarantined bool, prefix string) string {
	if includeQuarantined {
		return ""
	}
	return andFragment + "COALESCE(" + prefix + "quarantined, 0) = 0"
}

// filterClauses joins the validity, quarantine, scope, kind, and KB filters every recall
// query applies; prefix qualifies columns for joins and v must match it.
func filterClauses(in RecallInput, prefix string, v validityClauses) (string, []any) {
	validity, args := validityClause(in, v)
	quar := quarantineClause(in.IncludeQuarantined, prefix)
	scope, scopeArgs := scopeClauseFor(in, prefix)
	kind, kindArgs := kindClause(in.Kinds, prefix)
	refs, refsArgs := refsClause(in.Refs, prefix)
	tags, tagsArgs := tagsClause(in.Tags, prefix)
	actor, actorArgs := actorClause(in.Actor, prefix)
	tr, trArgs := timeRangeClause(in.Since, in.Until, prefix)
	ver := verifiedClause(in.VerifiedOnly, prefix)

	args = append(args, scopeArgs...)
	args = append(args, kindArgs...)
	args = append(args, refsArgs...)
	args = append(args, tagsArgs...)
	args = append(args, actorArgs...)
	args = append(args, trArgs...)
	return validity + quar + scope + kind + refs + tags + actor + tr + ver, args
}

func listActiveEntries(in RecallInput) ([]Entry, error) {
	fullTier2 := in.Expand || len(in.Refs) > 0
	clause, args := filterClauses(in, "", entryValidity)
	q := "SELECT " + entrySelectCols(fullTier2) + " FROM structured_memory WHERE 1=1" + clause + orderByAccessCreatedLimitClause
	return queryEntries(q, append(args, in.Limit*2)...)
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
		refs, _ := invalidateConflicting(KindFact, sc, in.SessionID, in.ProjectPath, in.Subject, pred, "")
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
