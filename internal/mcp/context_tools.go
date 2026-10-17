package mcp

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

func handleStoreContext(toolArgs map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	content, _ := toolArgs["content"].(string)
	label, _ := toolArgs["label"].(string)
	projectPath, _ := toolArgs["project_path"].(string)
	tags := toolArgs["tags"]
	kind, _ := toolArgs["kind"].(string)
	metadata := map[string]interface{}{}
	if raw, ok := toolArgs["metadata"].(map[string]interface{}); ok {
		metadata = raw
	}
	res, err := contextnotes.Store(sessionID, content, label, projectPath, tags, kind, metadata, emb)
	if err != nil {
		out := contextnotes.LimitErrorMap(err)
		resultJSON, _ := json.Marshal(out)
		errMsg := err.Error()
		if e, ok := out["error"].(string); ok {
			errMsg = e
		}
		logToolQuery("store_context", args, len(resultJSON), db.EstimateTokens(content), 0, context.SavingsMeta{}, start, cpuStart, projectPath, errMsg)
		return out
	}
	out := map[string]interface{}{
		"ref":                   res.Ref,
		"label":                 res.Label,
		"virtual_tokens_stored": res.VirtualTokensStored,
		"session_id":            res.SessionID,
		"stats":                 res.Stats,
	}
	if kind != "" {
		out["kind"] = kind
	} else if hasTag, _ := toolArgs["tags"].(string); hasTag != "" && strings.Contains(hasTag, contextnotes.TagKvRepair) {
		out["kind"] = contextnotes.KindKvRepair
	}
	if len(res.EvictedRefs) > 0 {
		out["evicted_refs"] = res.EvictedRefs
	}
	if extract, _ := toolArgs["extract_memory"].(bool); extract {
		ex := memory.ExtractFromText(content)
		stored, _ := memory.StoreExtracted(sessionID, projectPath, res.Ref, ex, memory.ScopeSession)
		if len(stored) > 0 {
			lines := make([]map[string]interface{}, 0, len(stored))
			for _, s := range stored {
				lines = append(lines, map[string]interface{}{"ref": s.Ref, "kind": s.Kind, "line": s.Line})
				if emb != nil {
					go memory.EmbedEntry(s.Ref, sessionID, s.Line, emb)
				}
			}
			out["memory_extracted"] = lines
		}
		if len(ex.Skipped) > 0 {
			out["memory_skipped"] = ex.Skipped
		}
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("store_context", args, len(resultJSON), db.EstimateTokens(content), 0, context.SavingsMeta{TokensSaved: res.VirtualTokensStored}, start, cpuStart, projectPath, "")
	return out
}

func handleFetchContext(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	repairReason, _ := toolArgs["repair_reason"].(string)
	refs := toolArgs["refs"]
	if refs == nil {
		refs, _ = toolArgs["ref"].(string)
	}
	res, err := contextnotes.Fetch(refs, sessionID, repairReason)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("fetch_context", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	tokensReturned := 0
	if v, ok := res.Stats["virtual_tokens_returned"].(int); ok {
		tokensReturned = v
	}
	out := map[string]interface{}{"notes": res.Notes, "stats": res.Stats}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("fetch_context", args, len(resultJSON), 0, tokensReturned, context.SavingsMeta{TokensUsed: tokensReturned}, start, cpuStart, projectPath, "")
	return out
}

func handleListContext(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	limit := 20
	if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
		limit = int(l)
	}
	if limit > 50 {
		limit = 50 // hard cap
	}
	expand := boolArg(toolArgs, "expand")
	res, err := contextnotes.List(sessionID, projectPath, limit)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("list_context", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	// MEM-10 & MEM-11: default to one line per entry under a hard cap unless expand=true
	// gated by feature_memory_v2.
	v2 := flags.Enabled(flags.KeyMemoryV2)
	notes := res.Notes
	out := map[string]interface{}{"total": res.Total}
	if v2 && !expand {
		var lines []map[string]interface{}
		for _, n := range notes {
			label := n.Label
			if label == "" {
				label = n.Kind
			}
			line := fmt.Sprintf("[%s] %s", n.Ref, label)
			lines = append(lines, map[string]interface{}{
				"ref":            n.Ref,
				"line":           line,
				"label":          n.Label,
				"kind":           n.Kind,
				"virtual_tokens": n.TokenEst,
			})
		}
		out["lines"] = lines
	} else {
		out["notes"] = notes
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("list_context", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, "")
	return out
}

func handleSearchContext(toolArgs map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	query, _ := toolArgs["query"].(string)
	sessionID, _ := toolArgs["session_id"].(string)
	limit := 5
	if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
		limit = int(l)
	}
	res, err := contextnotes.Search(query, sessionID, projectPath, limit, emb)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("search_context", args, len(resultJSON), db.EstimateTokens(query), 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	tokensReturned := 0
	if v, ok := res.Stats["virtual_tokens_returned"].(int); ok {
		tokensReturned = v
	}
	out := map[string]interface{}{"notes": res.Notes, "stats": res.Stats}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("search_context", args, len(resultJSON), db.EstimateTokens(query), tokensReturned, context.SavingsMeta{TokensUsed: tokensReturned}, start, cpuStart, projectPath, "")
	return out
}

func handleFlushContext(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	all, _ := toolArgs["all"].(bool)
	refs := toolArgs["refs"]
	if refs == nil {
		if r, ok := toolArgs["ref"].(string); ok {
			refs = r
		}
	}
	res, err := contextnotes.Flush(sessionID, refs, projectPath, all)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("flush_context", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	out := map[string]interface{}{
		"flushed_refs":         res.FlushedRefs,
		"virtual_tokens_freed": res.VirtualTokensFreed,
		"scope":                res.Scope,
		"stats":                res.Stats,
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("flush_context", args, len(resultJSON), res.VirtualTokensFreed, 0, context.SavingsMeta{FileBaseline: res.VirtualTokensFreed}, start, cpuStart, projectPath, "")
	return out
}

func handleReportKvRepairEvent(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	reason, _ := toolArgs["reason"].(string)
	outcome, _ := toolArgs["outcome"].(string)
	sessionID, _ := toolArgs["session_id"].(string)
	ref, _ := toolArgs["ref"].(string)
	modelID, _ := toolArgs["model_id"].(string)
	kvQuant, _ := toolArgs["kv_quant"].(string)
	detail, _ := toolArgs["detail"].(string)
	if pp, ok := toolArgs["project_path"].(string); ok && pp != "" {
		projectPath = pp
	}
	tokenEst := 0
	if v, ok := toolArgs["token_est"].(float64); ok {
		tokenEst = int(v)
	}
	err := contextnotes.ReportKvRepairEvent(contextnotes.ReportEventInput{
		Reason: reason, Outcome: outcome, SessionID: sessionID, ProjectPath: projectPath,
		Ref: ref, ModelID: modelID, KvQuant: kvQuant, TokenEst: tokenEst, Detail: detail,
	})
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("report_kv_repair_event", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	stats := map[string]interface{}{}
	contextnotes.AppendKvRepairStatsPublic(stats, projectPath)
	out := map[string]interface{}{"status": "ok", "reason": reason, "stats": stats}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("report_kv_repair_event", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, "")
	return out
}

func handleGetPinnedNote(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	pinnedProjectPath, _ := toolArgs["project_path"].(string)
	if projectPath != "" {
		pinnedProjectPath = projectPath
	}
	note, err := contextnotes.Pinned(pinnedProjectPath)
	if err != nil {
		return map[string]string{"error": err.Error()}
	}
	return map[string]interface{}{
		"ref":          note.Ref,
		"project_path": note.ProjectPath,
		"content":      note.Content,
		"token_est":    note.TokenEst,
		"created_at":   note.CreatedAt,
	}
}

func handleContextTool(toolName string, toolArgs map[string]interface{}, args map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, projectPath string) (interface{}, bool, error) {
	switch toolName {
	case "store_context":
		return handleStoreContext(toolArgs, emb, start, cpuStart, args), true, nil
	case "edit_context":
		return handleEditContext(toolArgs, emb, start, cpuStart, args, projectPath), true, nil
	case "define_context_fn":
		return handleDefineContextFn(toolArgs, args, start, cpuStart, projectPath), true, nil
	case "apply_context_fn":
		return handleApplyContextFn(toolArgs, emb, args, start, cpuStart, projectPath), true, nil
	case "list_context_fns":
		return handleListContextFns(toolArgs, args, start, cpuStart, projectPath), true, nil
	case "fetch_context":
		return handleFetchContext(toolArgs, start, cpuStart, args, projectPath), true, nil
	case "list_context":
		return handleListContext(toolArgs, start, cpuStart, args, projectPath), true, nil
	case "search_context":
		return handleSearchContext(toolArgs, emb, start, cpuStart, args, projectPath), true, nil
	case "flush_context":
		return handleFlushContext(toolArgs, start, cpuStart, args, projectPath), true, nil
	case "report_kv_repair_event":
		return handleReportKvRepairEvent(toolArgs, start, cpuStart, args, projectPath), true, nil
	case "get_pinned_note":
		return handleGetPinnedNote(toolArgs, start, cpuStart, args, projectPath), true, nil
	default:
		return nil, false, errs.New("not a context tool")
	}
}
