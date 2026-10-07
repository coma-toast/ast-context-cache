package mcp

import (
	"encoding/json"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// Reusable context functions: the model-defined half of Context Language Models
// (arXiv 2609.37725). Gated behind feature_context_fn, default off.

func defineContextFnTool() Tool {
	return Tool{
		Name: "define_context_fn",
		Description: "Define a reusable context function: a named (pattern, replacement) transform over stored ctx_* notes, saved so you can re-invoke it across a whole session instead of re-deriving the same edit every turn. " +
			"Go RE2 pattern (no backreferences or lookaround); $1 group references work in the replacement. " +
			"The pattern is compiled here, so a function that can never match is rejected at definition time rather than at apply time. " +
			"Redefining an existing name bumps version and resets its counters, so call_count measures the definition in force. " +
			"Pass expect_version when several agents share a name: the define fails with revision_conflict instead of clobbering another definition. " +
			"This stores a transform, never executable code — the agent cannot put code in context.db.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name":           map[string]string{"type": "string", "description": "Function name: 1-64 chars of letters, digits, underscore, dash, dot (required)"},
				"pattern":        map[string]string{"type": "string", "description": "Go RE2 regex selecting regions in each note's body (required)"},
				"replacement":    map[string]string{"type": "string", "description": "Replacement text; empty string deletes the matched regions. Supports $1 group references"},
				"description":    map[string]string{"type": "string", "description": "What the function does, for auditability"},
				"project_path":   map[string]string{"type": "string", "description": "Project scope for the function"},
				"session_id":     map[string]string{"type": "string", "description": "Session that authored the function"},
				"expect_version": map[string]string{"type": "integer", "description": "Fail with revision_conflict unless the stored function is still at this version"},
			},
			"required": []string{"name", "pattern"},
		},
		Tier: TierExtended,
	}
}

func applyContextFnTool() Tool {
	return Tool{
		Name: "apply_context_fn",
		Description: "Invoke a stored context function across notes — refs, or a whole session with session_id. " +
			"Each note is edited in place through the same revision log as edit_context, so a bad sweep is one edit_context(action=revert) per ref. " +
			"Returns per-ref match counts and token deltas plus a tokens_reclaimed total: the only way to tell whether a reusable function earned its storage. " +
			"Dry-run first with dry_run=true on anything wider than a couple of refs, and cap a single sweep with max_replacements.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name":             map[string]string{"type": "string", "description": "Function name to invoke (required)"},
				"refs":             map[string]string{"type": "string", "description": "Single ref, comma-separated list, or array of ctx_* refs"},
				"session_id":       map[string]string{"type": "string", "description": "Apply across every note in this session; omit when using refs"},
				"max_replacements": map[string]string{"type": "integer", "description": "Cap matches per note (default all, max 1000)"},
				"skip_errors":      map[string]string{"type": "boolean", "description": "Keep going after a note fails instead of stopping at the first error"},
				"dry_run":          map[string]string{"type": "boolean", "description": "Preview matches and token deltas without writing"},
			},
			"required": []string{"name"},
		},
		Tier: TierExtended,
	}
}

func listContextFnsTool() Tool {
	return Tool{
		Name: "list_context_fns",
		Description: "List stored context functions with their version, call_count, notes_touched and lifetime tokens_reclaimed, most-used first. " +
			"Patterns are omitted: list is for deciding which function to invoke, not for editing one. Returns retired functions too, so a name that stopped working can be traced.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"project_path": map[string]string{"type": "string", "description": "Project scope (default: unscoped functions)"},
				"limit":        map[string]string{"type": "integer", "description": "Max functions to return (default 20)"},
			},
		},
		Tier: TierCore,
	}
}

func fnRefsArg(toolArgs map[string]interface{}) []string {
	switch v := toolArgs["refs"].(type) {
	case string:
		return []string{v}
	case []interface{}:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return v
	}
	return nil
}

func handleDefineContextFn(toolArgs map[string]interface{}, args map[string]interface{}, start time.Time, cpuStart sys.CPUSample, projectPath string) interface{} {
	name, _ := toolArgs["name"].(string)
	pattern, _ := toolArgs["pattern"].(string)
	replacement, _ := toolArgs["replacement"].(string)
	description, _ := toolArgs["description"].(string)
	sessionID, _ := toolArgs["session_id"].(string)
	scope, _ := toolArgs["project_path"].(string)
	if scope == "" {
		scope = projectPath
	}

	fn, err := contextnotes.DefineFn(contextnotes.FnDefineInput{
		Name:          name,
		Description:   description,
		Pattern:       pattern,
		Replacement:   replacement,
		ProjectPath:   scope,
		SessionID:     sessionID,
		ExpectVersion: intArg(toolArgs, "expect_version"),
	})
	if err != nil {
		out := contextnotes.LimitErrorMap(err)
		resultJSON, _ := json.Marshal(out)
		errMsg := err.Error()
		if e, ok := out["error"].(string); ok {
			errMsg = e
		}
		logToolQuery("define_context_fn", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, errMsg)
		return out
	}
	resultJSON, _ := json.Marshal(fn)
	logToolQuery("define_context_fn", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, "")
	return map[string]interface{}{
		"fn":             fn.Name,
		"version":        fn.Version,
		"created":        fn.Version == 1,
		"project_path":   fn.ProjectPath,
		"session_id":     fn.SessionID,
		"pattern_bytes":  len(fn.Pattern),
		"functions_live": len(contextnotes.ListFns("", 1000)),
	}
}

func handleApplyContextFn(toolArgs map[string]interface{}, emb embedder.Interface, args map[string]interface{}, start time.Time, cpuStart sys.CPUSample, projectPath string) interface{} {
	name, _ := toolArgs["name"].(string)
	sessionID, _ := toolArgs["session_id"].(string)
	dryRun, _ := toolArgs["dry_run"].(bool)
	skipErrors, _ := toolArgs["skip_errors"].(bool)

	report, err := contextnotes.ApplyFn(contextnotes.FnApplyInput{
		Name:            name,
		Refs:            fnRefsArg(toolArgs),
		SessionID:       sessionID,
		MaxReplacements: intArg(toolArgs, "max_replacements"),
		DryRun:          dryRun,
		SkipErrors:      skipErrors,
	}, emb)
	if err != nil {
		out := contextnotes.LimitErrorMap(err)
		resultJSON, _ := json.Marshal(out)
		errMsg := err.Error()
		if e, ok := out["error"].(string); ok {
			errMsg = e
		}
		logToolQuery("apply_context_fn", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, errMsg)
		return out
	}
	resultJSON, _ := json.Marshal(report)
	savings := context.SavingsMeta{}
	if report.TokensReclaimed > 0 {
		savings = context.SavingsMeta{TokensSaved: report.TokensReclaimed}
	}
	logToolQuery("apply_context_fn", args, len(resultJSON), 0, 0, savings, start, cpuStart, projectPath, "")
	return report
}

func handleListContextFns(toolArgs map[string]interface{}, args map[string]interface{}, start time.Time, cpuStart sys.CPUSample, projectPath string) interface{} {
	scope, _ := toolArgs["project_path"].(string)
	if scope == "" {
		scope = projectPath
	}
	fns := contextnotes.ListFns(scope, intArg(toolArgs, "limit"))
	out := map[string]interface{}{"functions": fns, "count": len(fns)}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("list_context_fns", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, "")
	return out
}
