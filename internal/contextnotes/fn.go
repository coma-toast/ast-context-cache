package contextnotes

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// Reusable context functions are the paper's actual novelty (Context Language Models,
// arXiv 2609.37725): the model writes a function into its own context, then invokes it
// many times across a trace instead of re-deriving the same edit by hand each turn. In
// the paper that function is Python the model authors. Here it is a named
// (pattern, replacement) pair in SQLite, and every apply routes through Edit so the
// revision log, quotas, FTS reindex and vector hygiene are the same code path a manual
// edit takes. The agent never gets to store executable code in context.db.
//
// The reason this is a pattern/replacement pair and not a general library is the paper's
// own Discussion: a model-authored context transform is a prompt-injection surface, and
// the more expressive the transform, the more it can be used to rewrite context into
// something the author did not intend. Naming, versioning and dry-run auditing are the
// mitigations; expressiveness is deliberately not.

const (
	insertFnQuery = `INSERT INTO context_fns
			(name, project_path, session_id, description, pattern, replacement, pattern_hash, version)
		VALUES (?, ?, ?, ?, ?, ?, ?, 1)
		ON CONFLICT(name) DO UPDATE SET
			project_path = excluded.project_path,
			session_id = excluded.session_id,
			description = excluded.description,
			pattern = excluded.pattern,
			replacement = excluded.replacement,
			pattern_hash = excluded.pattern_hash,
			version = context_fns.version + 1,
			call_count = 0,
			notes_touched = 0,
			tokens_reclaimed = 0,
			updated_at = datetime('now'),
			retired_at = NULL`
	selectFnQuery = `SELECT name, COALESCE(project_path,''), COALESCE(session_id,''), COALESCE(description,''),
			pattern, replacement, pattern_hash, version, call_count, notes_touched, tokens_reclaimed, COALESCE(retired_at,'')
		FROM context_fns WHERE name = ?`
	selectFnMetaQuery = `SELECT name, COALESCE(project_path,''), COALESCE(description,''), version, call_count,
			notes_touched, tokens_reclaimed, COALESCE(retired_at,'')
		FROM context_fns WHERE project_path = ? OR COALESCE(project_path,'') = '' ORDER BY call_count DESC, name`
	countFnsQuery    = `SELECT COUNT(*) FROM context_fns WHERE retired_at IS NULL`
	retireFnQuery    = `UPDATE context_fns SET retired_at = datetime('now'), updated_at = datetime('now') WHERE name = ?`
	recordFnUseQuery = `UPDATE context_fns SET call_count = call_count + ?, notes_touched = notes_touched + ?, tokens_reclaimed = tokens_reclaimed + ? WHERE name = ?`
)

const (
	// maxFnPattern bounds the stored pattern so a function cannot be used to smuggle an
	// unbounded blob into every note it touches.
	maxFnPattern = 8192
	// defaultMaxFns bounds how many live functions one store can hold. Each is a
	// reusable transform an agent can apply at scale, so the registry needs the same
	// kind of cap as the note store.
	defaultMaxFns = 50
	// maxFnApplyTargets bounds one apply. A function applied across a whole session is
	// the point of the feature, so the ceiling is generous but not unbounded.
	maxFnApplyTargets = 200
)

// LoadMaxFns is how many live context functions the store holds.
func LoadMaxFns() int {
	n := db.SettingInt("context_max_fns", "AST_CONTEXT_MAX_FNS", defaultMaxFns)
	if n < 0 {
		n = 0
	}
	return n
}

// Fn is a stored, reusable context transform.
type Fn struct {
	Name            string `json:"name"`
	ProjectPath     string `json:"project_path,omitempty"`
	SessionID       string `json:"session_id,omitempty"`
	Description     string `json:"description,omitempty"`
	Pattern         string `json:"pattern,omitempty"`
	Replacement     string `json:"replacement,omitempty"`
	PatternHash     string `json:"-"`
	Version         int    `json:"version"`
	CallCount       int    `json:"call_count"`
	NotesTouched    int    `json:"notes_touched"`
	TokensReclaimed int    `json:"tokens_reclaimed"`
	Retired         bool   `json:"retired,omitempty"`
}

// FnDefineInput registers or replaces a named transform. Redefining an existing name
// bumps version and resets its counters, so call_count measures the definition actually
// in force rather than a lifetime total across unrelated rewrites.
type FnDefineInput struct {
	Name        string
	Description string
	Pattern     string
	Replacement string
	ProjectPath string
	SessionID   string
	// ExpectVersion guards against a concurrent agent redefining the same name. The
	// same lost-update problem edit_context's expect_revision solves for notes.
	ExpectVersion int
}

func validFnName(name string) bool {
	if len(name) < 1 || len(name) > 64 {
		return false
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		case r == '_', r == '-', r == '.':
		default:
			return false
		}
	}
	return true
}

// DefineFn registers a reusable context transform.
func DefineFn(in FnDefineInput) (*Fn, error) {
	name := strings.TrimSpace(in.Name)
	if !validFnName(name) {
		return nil, errs.NewCode(errs.CodeInvalidInput,
			"name must be 1-64 characters of letters, digits, underscore, dash, or dot")
	}
	pattern := in.Pattern
	if pattern == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "pattern required")
	}
	if len(pattern) > maxFnPattern {
		return nil, errs.NewCode(errs.CodeInvalidInput,
			fmt.Sprintf("pattern is %d bytes; the limit is %d", len(pattern), maxFnPattern))
	}
	// Compile at definition time. A function that can never match is dead weight the
	// agent pays to store and to rediscover on every apply.
	if _, err := regexp.Compile(pattern); err != nil {
		return nil, errs.NewCode(errs.CodeInvalidInput, "invalid pattern: "+err.Error())
	}
	if existing, ok := fnByName(name); ok {
		if in.ExpectVersion > 0 && in.ExpectVersion != existing.Version {
			return nil, errs.NewCode(errs.CodeConflict, "revision_conflict",
				"fn", name, "expected_version", in.ExpectVersion, "current_version", existing.Version,
				"detail", "another agent redefined this function; list_context_fns to re-read before redefining")
		}
	} else {
		var live int
		db.ContextDB.QueryRow(countFnsQuery).Scan(&live)
		if max := LoadMaxFns(); max > 0 && live >= max {
			// Same LimitError shape as a note-quota rejection, so the MCP handler's
			// existing context_limit_exceeded mapping covers it with no new branch.
			return nil, newLimitError("context_fns", live, max, 1)
		}
	}
	hash := search.ContentHash(pattern)
	if _, err := db.ContextDB.Exec(insertFnQuery, name, strings.TrimSpace(in.ProjectPath),
		strings.TrimSpace(in.SessionID), strings.TrimSpace(in.Description), pattern, in.Replacement, hash); err != nil {
		return nil, errs.WrapMessage("failed to store context function", err, "fn", name)
	}
	fn, ok := fnByName(name)
	if !ok {
		return nil, errs.NewCode(errs.CodeInternal, "context function vanished after write", "fn", name)
	}
	logger.Info("Defined context function", "fn", name, "version", fn.Version, "pattern_bytes", len(pattern))
	return &fn, nil
}

func fnByName(name string) (Fn, bool) {
	var f Fn
	var retired string
	err := db.ContextDB.QueryRow(selectFnQuery, name).Scan(&f.Name, &f.ProjectPath, &f.SessionID,
		&f.Description, &f.Pattern, &f.Replacement, &f.PatternHash, &f.Version, &f.CallCount,
		&f.NotesTouched, &f.TokensReclaimed, &retired)
	if err != nil {
		return Fn{}, false
	}
	f.Retired = strings.TrimSpace(retired) != ""
	return f, true
}

// ListFns returns function metadata (never patterns) for the project, most-used first.
func ListFns(projectPath string, limit int) []Fn {
	projectPath = strings.TrimSpace(projectPath)
	if limit <= 0 {
		limit = 20
	}
	rows, err := db.ContextDB.Query(selectFnMetaQuery+" LIMIT ?", projectPath, limit)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []Fn
	for rows.Next() {
		var f Fn
		var retired string
		if rows.Scan(&f.Name, &f.ProjectPath, &f.Description, &f.Version, &f.CallCount,
			&f.NotesTouched, &f.TokensReclaimed, &retired) == nil {
			f.Retired = strings.TrimSpace(retired) != ""
			out = append(out, f)
		}
	}
	return out
}

// RetireFn tombstones a function. It is not deleted, so a name's history stays
// auditable and a later define with the same name starts from a clean counter.
func RetireFn(name string) (bool, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return false, errs.NewCode(errs.CodeInvalidInput, "name required")
	}
	if _, ok := fnByName(name); !ok {
		return false, errs.NewCode(errs.CodeNotFound, "no such context function: "+name)
	}
	if _, err := db.ContextDB.Exec(retireFnQuery, name); err != nil {
		return false, errs.WrapMessage("failed to retire context function", err, "fn", name)
	}
	return true, nil
}

// FnApplyInput invokes a stored function across notes. Refs and SessionID are the two
// ways to pick targets: Refs is explicit, SessionID sweeps a whole session, which is the
// case the paper's compact_turns-style function exists for.
type FnApplyInput struct {
	Name      string
	Refs      []string
	SessionID string
	// MaxReplacements caps matches per note, so one pathological pattern cannot rewrite
	// an entire session's context in a single call.
	MaxReplacements int
	DryRun          bool
	// SkipErrors keeps going when one note fails. Off by default: a partial apply that
	// half-succeeded is harder to reason about than one that stopped.
	SkipErrors bool
}

// FnApplyReport is what one apply cost or saved. TokensReclaimed is the same metric
// edit_context reports: the only evidence a reusable function earned its storage.
type FnApplyReport struct {
	Fn              string                 `json:"fn"`
	FnVersion       int                    `json:"fn_version"`
	Targets         int                    `json:"targets"`
	Changed         int                    `json:"changed"`
	Unchanged       int                    `json:"unchanged"`
	Skipped         int                    `json:"skipped"`
	Failed          int                    `json:"failed"`
	MatchedRegions  int                    `json:"matched_regions"`
	TokensBefore    int                    `json:"tokens_before"`
	TokensAfter     int                    `json:"tokens_after"`
	TokensReclaimed int                    `json:"tokens_reclaimed"`
	DryRun          bool                   `json:"dry_run,omitempty"`
	Refs            []FnApplyRefResult     `json:"refs"`
	Errors          []FnApplyError         `json:"errors,omitempty"`
	Stats           map[string]interface{} `json:"stats"`
}

// FnApplyRefResult is the per-note outcome.
type FnApplyRefResult struct {
	Ref              string `json:"ref"`
	Revision         int    `json:"revision"`
	PreviousRevision int    `json:"previous_revision"`
	Changed          bool   `json:"changed"`
	MatchedRegions   int    `json:"matched_regions"`
	TokensBefore     int    `json:"tokens_before"`
	TokensAfter      int    `json:"tokens_after"`
	TokensReclaimed  int    `json:"tokens_reclaimed"`
}

// FnApplyError records why one note was skipped without failing the whole apply.
type FnApplyError struct {
	Ref    string `json:"ref"`
	Reason string `json:"reason"`
}

// ApplyFn invokes a stored function across the selected notes. Each note is edited
// through Edit, so every apply lands in the revision log and is revertable per note:
// a bad sweep is one edit_context(action=revert) per ref, not a restore-from-backup.
func ApplyFn(in FnApplyInput, emb embedder.Interface) (*FnApplyReport, error) {
	name := strings.TrimSpace(in.Name)
	if name == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "name required")
	}
	fn, ok := fnByName(name)
	if !ok {
		return nil, errs.NewCode(errs.CodeNotFound, "no such context function: "+name)
	}
	if fn.Retired {
		return nil, errs.NewCode(errs.CodeConflict, "function_retired",
			"fn", name, "detail", "this function was retired; define_context_fn to bring it back")
	}
	targets, sessions, err := fnTargets(in)
	if err != nil {
		return nil, err
	}
	if len(targets) > maxFnApplyTargets {
		targets = targets[:maxFnApplyTargets]
	}

	report := &FnApplyReport{Fn: name, FnVersion: fn.Version, Targets: len(targets), DryRun: in.DryRun}
	for _, ref := range targets {
		res, err := Edit(EditInput{
			Action:          EditReplace,
			Ref:             ref,
			MaxReplacements: in.MaxReplacements,
			Pattern:         fn.Pattern,
			Replacement:     fn.Replacement,
			DryRun:          in.DryRun,
		}, emb)
		if err != nil {
			report.Failed++
			report.Errors = append(report.Errors, FnApplyError{Ref: ref, Reason: err.Error()})
			if !in.SkipErrors {
				report.Stats = BuildStatsBlock(sessions, LoadLimits())
				return report, nil
			}
			continue
		}
		entry := FnApplyRefResult{
			Ref: ref, Revision: res.Revision, PreviousRevision: res.PreviousRevision,
			Changed: res.Changed, MatchedRegions: res.MatchedRegions,
			TokensBefore: res.TokensBefore, TokensAfter: res.TokensAfter,
			TokensReclaimed: res.TokensReclaimed,
		}
		report.Refs = append(report.Refs, entry)
		report.MatchedRegions += res.MatchedRegions
		report.TokensBefore += res.TokensBefore
		report.TokensAfter += res.TokensAfter
		report.TokensReclaimed += res.TokensReclaimed
		if res.Changed {
			report.Changed++
		} else {
			report.Unchanged++
		}
	}
	// Counters only move for a real apply. A dry run that looks like it reclaimed
	// 40k tokens would make the registry's own metric a fiction.
	if !in.DryRun {
		if _, err := db.ContextDB.Exec(recordFnUseQuery, 1, report.Changed, report.TokensReclaimed, name); err != nil {
			logger.Warn("Failed to record context function usage", "fn", name, "error", err)
		}
	}
	report.Stats = BuildStatsBlock(sessions, LoadLimits())
	logger.Info("Applied context function", "fn", name, "version", fn.Version, "targets", report.Targets,
		"changed", report.Changed, "skipped", report.Failed, "tokens_reclaimed", report.TokensReclaimed,
		"dry_run", in.DryRun)
	return report, nil
}

// fnTargets resolves the notes to apply to and the session whose stats block applies.
func fnTargets(in FnApplyInput) ([]string, string, error) {
	refs := make([]string, 0, len(in.Refs))
	seen := map[string]bool{}
	for _, r := range in.Refs {
		for _, part := range strings.Split(r, ",") {
			part = strings.TrimSpace(part)
			if part == "" || seen[part] {
				continue
			}
			seen[part] = true
			refs = append(refs, part)
		}
	}
	sessionID := strings.TrimSpace(in.SessionID)
	if len(refs) == 0 {
		if sessionID == "" {
			return nil, "", errs.NewCode(errs.CodeInvalidInput, "refs or session_id required")
		}
		list, err := List(sessionID, "", maxFnApplyTargets)
		if err != nil {
			return nil, "", errs.WrapMessage("failed to list notes for context function apply", err)
		}
		for _, n := range list.Notes {
			if !seen[n.Ref] {
				seen[n.Ref] = true
				refs = append(refs, n.Ref)
			}
		}
	}
	if len(refs) == 0 {
		return nil, "", errs.NewCode(errs.CodeNotFound, "no notes to apply to")
	}
	return refs, sessionID, nil
}
