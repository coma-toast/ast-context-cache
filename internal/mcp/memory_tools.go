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
		// Project memories describe the repo, not the checkout, so a note taken in
		// one WTG worktree should come back in a sibling worktree on another branch.
		IncludeRepoSiblings: true,
	}
	if v, ok := toolArgs["repo_siblings"].(bool); ok {
		in.IncludeRepoSiblings = v
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
		Predicate:   strArg(toolArgs, "predicate"),
		All:         boolArg(toolArgs, "all"),
	}
	if scope, ok := toolArgs["scope"].(string); ok {
		in.Scope = memory.Scope(strings.ToLower(strings.TrimSpace(scope)))
	}
	refsRaw := toolArgs["refs"]
	if refsRaw == nil {
		refsRaw = toolArgs["ref"]
	}
	if refsRaw != nil {
		in.Refs = parseStringList(refsRaw)
	}
	res, err := memory.Forget(in)
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("forget_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
		return out
	}
	out := map[string]interface{}{
		"invalidated_refs":     res.InvalidatedRefs,
		"virtual_tokens_freed": res.VirtualTokensFreed,
	}
	if len(in.Refs) > 0 {
		out["invalidated"] = nonNil(res.Invalidated)
		if len(res.NotFound) > 0 {
			out["not_found"] = res.NotFound
		}
		if len(res.AlreadyInvalid) > 0 {
			out["already_invalid"] = res.AlreadyInvalid
		}
		if len(res.ScopeMismatch) > 0 {
			out["scope_mismatch"] = res.ScopeMismatch
		}
		if res.InvalidatedRefs == 0 && len(res.AlreadyInvalid) == 0 {
			msg := "no active mem_* entry matched the given refs"
			if len(res.ScopeMismatch) > 0 {
				msg += " (scope_mismatch refs are outside the given scope; omit scope to resolve it from each ref)"
			}
			out["error"] = msg
		}
	}
	errMsg, _ := out["error"].(string)
	resultJSON, _ := json.Marshal(out)
	logToolQuery("forget_memory", args, len(resultJSON), res.VirtualTokensFreed, 0, context.SavingsMeta{FileBaseline: res.VirtualTokensFreed}, start, cpuStart, pp, errMsg)
	return out
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
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

// parseStringList accepts a JSON array, a single string, a comma-separated
// string ("mem_a,mem_b"), or a string holding a JSON array ("[\"mem_a\"]",
// which some clients send when the schema type is string).
func parseStringList(raw interface{}) []string {
	var out []string
	add := func(s string) {
		for _, part := range strings.Split(s, ",") {
			if p := strings.Trim(strings.TrimSpace(part), `"'`); p != "" {
				out = append(out, p)
			}
		}
	}
	switch v := raw.(type) {
	case string:
		t := strings.TrimSpace(v)
		if strings.HasPrefix(t, "[") {
			var arr []string
			if json.Unmarshal([]byte(t), &arr) == nil {
				for _, s := range arr {
					add(s)
				}
				return out
			}
			t = strings.TrimSuffix(strings.TrimPrefix(t, "["), "]")
		}
		add(t)
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				add(s)
			}
		}
	case []string:
		for _, s := range v {
			add(s)
		}
	}
	return out
}
