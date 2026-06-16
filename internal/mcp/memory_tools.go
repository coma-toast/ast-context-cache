package mcp

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

func handleMemoryTool(toolName string, toolArgs map[string]interface{}, args map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, projectPath string) (interface{}, bool, error) {
	switch toolName {
	case "store_memory":
		return handleStoreMemory(toolArgs, emb, start, cpuStart, args, projectPath), true, nil
	case "recall_memory":
		return handleRecallMemory(toolArgs, emb, start, cpuStart, args, projectPath), true, nil
	case "forget_memory":
		return handleForgetMemory(toolArgs, start, cpuStart, args, projectPath), true, nil
	default:
		return nil, false, nil
	}
}

func handleStoreMemory(toolArgs map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	kindStr, _ := toolArgs["kind"].(string)
	scopeStr, _ := toolArgs["scope"].(string)
	pp, _ := toolArgs["project_path"].(string)
	if pp == "" {
		pp = projectPath
	}
	in := memory.StoreInput{
		Kind:        memory.Kind(strings.ToLower(strings.TrimSpace(kindStr))),
		Scope:       memory.Scope(strings.ToLower(strings.TrimSpace(scopeStr))),
		SessionID:   sessionID,
		ProjectPath: pp,
		Subject:     strArg(toolArgs, "subject"),
		Predicate:   strArg(toolArgs, "predicate"),
		Object:      strArg(toolArgs, "object"),
		Rule:        strArg(toolArgs, "rule"),
		SourceRef:   strArg(toolArgs, "source_ref"),
	}
	if v, ok := toolArgs["invalidate_previous"].(bool); ok {
		in.InvalidatePrevious = v
	}
	res, err := memory.Store(in)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("store_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
		return out
	}
	if emb != nil {
		go memory.EmbedEntry(res.Ref, sessionID, res.Line, emb)
	}
	out := map[string]interface{}{
		"ref":                   res.Ref,
		"kind":                  res.Kind,
		"line":                  res.Line,
		"virtual_tokens_stored": res.VirtualTokensStored,
		"invalidated_refs":      res.InvalidatedRefs,
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("store_memory", args, len(resultJSON), res.VirtualTokensStored, 0, context.SavingsMeta{TokensSaved: res.VirtualTokensStored}, start, cpuStart, pp, "")
	return out
}

func handleRecallMemory(toolArgs map[string]interface{}, emb embedder.Interface, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	pp, _ := toolArgs["project_path"].(string)
	if pp == "" {
		pp = projectPath
	}
	limit := 10
	if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
		limit = int(l)
	}
	budget := 800
	if b, ok := toolArgs["token_budget"].(float64); ok && b > 0 {
		budget = int(b)
	}
	in := memory.RecallInput{
		Query:       strArg(toolArgs, "query"),
		SessionID:   sessionID,
		ProjectPath: pp,
		AsOf:        strArg(toolArgs, "as_of"),
		Limit:       limit,
		TokenBudget: budget,
	}
	if k, ok := toolArgs["kind"].(string); ok && k != "" {
		in.Kinds = []memory.Kind{memory.Kind(strings.ToLower(k))}
	}
	if kinds, ok := toolArgs["kinds"].([]interface{}); ok {
		for _, item := range kinds {
			if s, ok := item.(string); ok {
				in.Kinds = append(in.Kinds, memory.Kind(strings.ToLower(s)))
			}
		}
	}
	if scope, ok := toolArgs["scope"].(string); ok && scope != "" {
		in.Scope = memory.Scope(strings.ToLower(scope))
	}
	res, err := memory.Recall(in, emb)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("recall_memory", args, len(resultJSON), db.EstimateTokens(in.Query), 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
		return out
	}
	out := map[string]interface{}{
		"lines":            res.Lines,
		"formatted":        res.Formatted,
		"tokens_used":      res.TokensUsed,
		"tokens_saved_est": res.TokensSavedEst,
		"refs_accessed":    res.RefsAccessed,
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("recall_memory", args, len(resultJSON), db.EstimateTokens(in.Query), res.TokensUsed, context.SavingsMeta{TokensUsed: res.TokensUsed, TokensSaved: res.TokensSavedEst}, start, cpuStart, pp, "")
	return out
}

func handleForgetMemory(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	sessionID, _ := toolArgs["session_id"].(string)
	pp, _ := toolArgs["project_path"].(string)
	if pp == "" {
		pp = projectPath
	}
	in := memory.ForgetInput{
		SessionID:   sessionID,
		ProjectPath: pp,
		Subject:     strArg(toolArgs, "subject"),
		Predicate: strArg(toolArgs, "predicate"),
		All:       boolArg(toolArgs, "all"),
	}
	if scope, ok := toolArgs["scope"].(string); ok {
		in.Scope = memory.Scope(strings.ToLower(scope))
	}
	if refs := toolArgs["refs"]; refs != nil {
		in.Refs = parseStringList(refs)
	}
	res, err := memory.Forget(in)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("forget_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
		return out
	}
	out := map[string]interface{}{
		"invalidated_refs":       res.InvalidatedRefs,
		"virtual_tokens_freed": res.VirtualTokensFreed,
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("forget_memory", args, len(resultJSON), res.VirtualTokensFreed, 0, context.SavingsMeta{FileBaseline: res.VirtualTokensFreed}, start, cpuStart, pp, "")
	return out
}

func strArg(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func boolArg(m map[string]interface{}, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}

func parseStringList(raw interface{}) []string {
	switch v := raw.(type) {
	case string:
		if strings.TrimSpace(v) != "" {
			return []string{strings.TrimSpace(v)}
		}
	case []interface{}:
		var out []string
		for _, item := range v {
			if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	case []string:
		var out []string
		for _, s := range v {
			if strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	}
	return nil
}
