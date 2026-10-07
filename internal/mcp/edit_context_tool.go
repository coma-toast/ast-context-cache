package mcp

import (
	"encoding/json"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// editContextTool is the context-as-a-file primitive: stored virtual context is
// normally write-once, so changing a note meant flushing it (invalidating the
// ctx_* stub already in chat) and storing a new one. This lets the agent rewrite
// a note in place and keep its ref.
func editContextTool() Tool {
	return Tool{
		Name: "edit_context",
		Description: "Edit stored virtual context in place, keeping the same ctx_* ref — the context-as-a-file primitive (Context Language Models, arXiv 2609.37725). " +
			"append: add to the end. rewrite: replace the whole body. replace: surgical edit of matched regions. delete: drop matched regions. revert: restore a prior revision. " +
			"replace and delete address regions by regex pattern OR by start_line/end_line (1-indexed, inclusive), never both; an empty replacement deletes the matched text. " +
			"Every edit is reversible and returns tokens_before/tokens_after/tokens_reclaimed — the token analogue of the paper's prefix-reuse FLOPs, the only way to tell whether a context-management edit paid for itself. " +
			"Pass expect_revision when several agents share a note: the edit fails with revision_conflict instead of clobbering someone else's change. " +
			"Use dry_run to preview matches and token deltas first.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"action":           map[string]string{"type": "string", "description": "append, replace, delete, rewrite, or revert (required)"},
				"ref":              map[string]string{"type": "string", "description": "ctx_* ref of the note to edit (required)"},
				"session_id":       map[string]string{"type": "string", "description": "If set, reject refs owned by another session"},
				"content":          map[string]string{"type": "string", "description": "Text for append or rewrite"},
				"pattern":          map[string]string{"type": "string", "description": "Regex (Go RE2: no backreferences or lookaround) selecting regions to replace or delete"},
				"replacement":      map[string]string{"type": "string", "description": "Replacement text for replace; empty string deletes the match. Supports $1 group references"},
				"start_line":       map[string]string{"type": "integer", "description": "First line of a region to replace or delete (1-indexed, inclusive)"},
				"end_line":         map[string]string{"type": "integer", "description": "Last line of that region (1-indexed, inclusive)"},
				"max_replacements": map[string]string{"type": "integer", "description": "Cap how many pattern matches are touched (default all, max 1000)"},
				"to_revision":      map[string]string{"type": "integer", "description": "Revision to restore for revert (default: the most recent superseded body)"},
				"expect_revision":  map[string]string{"type": "integer", "description": "Fail with revision_conflict unless the note is still at this revision"},
				"dry_run":          map[string]string{"type": "boolean", "description": "Report matched regions and token deltas without writing"},
			},
			"required": []string{"action", "ref"},
		},
		Tier: TierExtended,
	}
}

func handleEditContext(toolArgs map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	content, _ := toolArgs["content"].(string)
	pattern, _ := toolArgs["pattern"].(string)
	replacement, _ := toolArgs["replacement"].(string)
	action, _ := toolArgs["action"].(string)
	ref, _ := toolArgs["ref"].(string)
	dryRun, _ := toolArgs["dry_run"].(bool)

	in := contextnotes.EditInput{
		Action:      action,
		Ref:         ref,
		SessionID:   sessionID,
		Content:     content,
		Pattern:     pattern,
		Replacement: replacement,
		DryRun:      dryRun,
	}
	in.ExpectRevision = intArg(toolArgs, "expect_revision")
	in.MaxReplacements = intArg(toolArgs, "max_replacements")
	in.ToRevision = intArg(toolArgs, "to_revision")
	in.StartLine = intArg(toolArgs, "start_line")
	in.EndLine = intArg(toolArgs, "end_line")

	res, err := contextnotes.Edit(in, emb)
	if err != nil {
		out := contextnotes.LimitErrorMap(err)
		resultJSON, _ := json.Marshal(out)
		errMsg := err.Error()
		if e, ok := out["error"].(string); ok {
			errMsg = e
		}
		logToolQuery("edit_context", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, errMsg)
		return out
	}
	resultJSON, _ := json.Marshal(res)
	savings := context.SavingsMeta{}
	if res.TokensReclaimed > 0 {
		savings = context.SavingsMeta{TokensSaved: res.TokensReclaimed}
	}
	logToolQuery("edit_context", args, len(resultJSON), 0, 0, savings, start, cpuStart, projectPath, "")
	return map[string]interface{}{
		"ref":               res.Ref,
		"action":            res.Action,
		"revision":          res.Revision,
		"previous_revision": res.PreviousRevision,
		"changed":           res.Changed,
		"matched_regions":   res.MatchedRegions,
		"tokens_before":     res.TokensBefore,
		"tokens_after":      res.TokensAfter,
		"tokens_reclaimed":  res.TokensReclaimed,
		"dry_run":           res.DryRun,
		"session_id":        res.SessionID,
		"stats":             res.Stats,
	}
}
