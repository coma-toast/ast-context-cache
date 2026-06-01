package mcp

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

func handleStoreContext(toolArgs map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	content, _ := toolArgs["content"].(string)
	label, _ := toolArgs["label"].(string)
	projectPath, _ := toolArgs["project_path"].(string)
	tags := toolArgs["tags"]
	res, err := contextnotes.Store(sessionID, content, label, projectPath, tags, emb)
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
	if len(res.EvictedRefs) > 0 {
		out["evicted_refs"] = res.EvictedRefs
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("store_context", args, len(resultJSON), db.EstimateTokens(content), 0, context.SavingsMeta{TokensSaved: res.VirtualTokensStored}, start, cpuStart, projectPath, "")
	return out
}

func handleFetchContext(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	refs := toolArgs["refs"]
	if refs == nil {
		refs, _ = toolArgs["ref"].(string)
	}
	res, err := contextnotes.Fetch(refs, sessionID)
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
	res, err := contextnotes.List(sessionID, projectPath, limit)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("list_context", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	out := map[string]interface{}{"notes": res.Notes, "total": res.Total}
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

func handleContextTool(toolName string, toolArgs map[string]interface{}, args map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, projectPath string) (interface{}, bool, error) {
	switch toolName {
	case "store_context":
		return handleStoreContext(toolArgs, emb, start, cpuStart, args), true, nil
	case "fetch_context":
		return handleFetchContext(toolArgs, start, cpuStart, args, projectPath), true, nil
	case "list_context":
		return handleListContext(toolArgs, start, cpuStart, args, projectPath), true, nil
	case "search_context":
		return handleSearchContext(toolArgs, emb, start, cpuStart, args, projectPath), true, nil
	case "flush_context":
		return handleFlushContext(toolArgs, start, cpuStart, args, projectPath), true, nil
	default:
		return nil, false, errors.New("not a context tool")
	}
}
