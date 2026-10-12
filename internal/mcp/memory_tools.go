package mcp

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// parseTTL converts a TTL string ("Nd", "Nh", "Nm", or plain days) into days, hours
// and minutes so that hour and minute precision is preserved rather than rounded up to
// a full day (MEM-8). Returns an error for unknown units using the errs package.
func parseTTL(s string) (days, hours, minutes int, err error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, 0, 0, nil
	}
	// Plain integer means days (backward compat).
	if parsed, err := strconv.Atoi(s); err == nil {
		return parsed, 0, 0, nil
	}
	// time.ParseDuration handles standard units "h", "m", "s", "ms", "us", "ns" and
	// combinations like "1h30m". Go does NOT support "d" in ParseDuration.
	if d, err := time.ParseDuration(s); err == nil {
		hours = int(d / time.Hour)
		minutes = int(d/time.Minute) % 60
		return 0, hours, minutes, nil
	}
	// Handle "Nd" suffix and combinations like "2d3h", "1d6h30m".
	if idx := strings.Index(s, "d"); idx >= 0 || strings.Index(s, "D") >= 0 {
		dIdx := strings.Index(s, "d")
		if dIdx < 0 {
			dIdx = strings.Index(s, "D")
		}
		dayStr := s[:dIdx]
		if n, err := strconv.Atoi(dayStr); err == nil {
			days = n
			remainder := s[dIdx+1:]
			if remainder != "" {
				if d, err := time.ParseDuration(remainder); err == nil {
					hours += int(d / time.Hour)
					minutes += int(d/time.Minute) % 60
				} else {
					return 0, 0, 0, errs.NewCode(errs.CodeInvalidInput, "invalid ttl: %s", s)
				}
			}
			return days, hours, minutes, nil
		}
	}
	return 0, 0, 0, errs.NewCode(errs.CodeInvalidInput, "invalid ttl: %s", s)
}

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
		// A new fact supersedes the active one for its subject and predicate unless the
		// caller opts out.
		InvalidatePrevious: true,
	}
	ttlDays, ttlHours, ttlMinutes, err := parseTTL(strArg(toolArgs, "ttl"))
	if err != nil {
		logToolQuery("store_memory", args, 0, 0, 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
		return map[string]string{"error": err.Error()}
	}
	if ttlDays > 0 {
		in.TTL = ttlDays
	}
	if ttlHours > 0 {
		in.TTLHours = ttlHours
	}
	if ttlMinutes > 0 {
		in.TTLMinutes = ttlMinutes
	}
	if expiresAt := strArg(toolArgs, "expires_at"); expiresAt != "" {
		in.ExpiresAt = expiresAt
	}
	// MEM-5: optional world-time anchors: the fact was/is true in the world during
	// [valid_at, invalid_at]. When absent world time equals system time.
	if v := strArg(toolArgs, "valid_at"); v != "" {
		in.WorldValidAt = v
	}
	if v := strArg(toolArgs, "invalid_at"); v != "" {
		in.WorldInvalidAt = v
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
		go func() {
			_, _ = memory.EmbedEntryWithVector(res.Ref, sessionID, res.Line, emb)
		}()
	}

	out := map[string]interface{}{
		"ref":                   res.Ref,
		"kind":                  res.Kind,
		"line":                  res.Line,
		"virtual_tokens_stored": res.VirtualTokensStored,
		"invalidated_refs":      res.InvalidatedRefs,
	}

	if in.Kind == memory.KindFact {
		entry := memory.Entry{
			Ref:            res.Ref,
			Kind:           in.Kind,
			Scope:          in.Scope,
			SessionID:      sessionID,
			ProjectPath:    in.ProjectPath,
			Subject:        in.Subject,
			Predicate:      in.Predicate,
			Object:         in.Object,
			Rule:           in.Rule,
			SourceRef:      in.SourceRef,
			TokenEst:       res.VirtualTokensStored,
			AccessCount:    1,
			LastAccessedAt: time.Now().UTC().Format("2006-01-02 15:04:05"),
			CreatedAt:      time.Now().UTC().Format("2006-01-02 15:04:05"),
		}
		conflicts, method := memory.ConflictCandidates(entry, emb, 300*time.Millisecond)
		if len(conflicts) > 0 {
			var confSlice []map[string]interface{}
			for _, c := range conflicts {
				confSlice = append(confSlice, map[string]interface{}{
					"ref":        c.Ref,
					"fact":       c.Line,
					"similarity": c.Similarity,
				})
			}
			out["possible_conflicts"] = confSlice
			out["conflict_method"] = method
		}
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
	asOf := strArg(toolArgs, "as_of")
	if asOf != "" {
		normalized, err := db.NormalizeSQLTime(asOf)
		if err != nil {
			out := map[string]string{"error": err.Error()}
			resultJSON, _ := json.Marshal(out)
			logToolQuery("recall_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
			return out
		}
		asOf = normalized
	}
	in := memory.RecallInput{
		Query:       strArg(toolArgs, "query"),
		SessionID:   sessionID,
		ProjectPath: pp,
		AsOf:        asOf,
		// MEM-5: as_of_system=true asks what was known at that time using system
		// valid_from/valid_until instead of world time.
		AsOfSystem:  boolArg(toolArgs, "as_of_system"),
		Limit:       limit,
		TokenBudget: budget,
		// MEM-9: debug=true lists entries that matched but were demoted below the
		// budget, with their scores and the reason.
		Debug: boolArg(toolArgs, "debug"),
		// MEM-10: expand=true returns full bodies instead of compact one-line
		// summaries (the default).
		Expand: boolArg(toolArgs, "expand"),
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
	if v, ok := toolArgs["history"].(bool); ok && v {
		return recallHistory(toolArgs, start, cpuStart, args, pp)
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
	if len(res.Demoted) > 0 {
		out["demoted"] = res.Demoted
	}
	if in.Expand {
		out["entries"] = res.Entries
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("recall_memory", args, len(resultJSON), db.EstimateTokens(in.Query), res.TokensUsed, context.SavingsMeta{TokensUsed: res.TokensUsed, TokensSaved: res.TokensSavedEst}, start, cpuStart, pp, "")
	return out
}

// recallHistory handles recall_memory with history=true, returning the full version chain.
func recallHistory(toolArgs map[string]interface{}, start time.Time, cpuStart sys.CPUSample, args map[string]interface{}, projectPath string) interface{} {
	ref := strArg(toolArgs, "ref")
	query := strArg(toolArgs, "query")
	includeWorld := boolArg(toolArgs, "include_world")
	if ref == "" && query == "" {
		out := map[string]string{"error": "history requires either a ref or a query"}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("recall_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, "")
		return out
	}
	hist, err := memory.History(memory.HistoryInput{
		Ref:          ref,
		Query:        query,
		IncludeWorld: includeWorld,
	})
	if err != nil {
		out := map[string]string{"error": err.Error()}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("recall_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, projectPath, err.Error())
		return out
	}
	out := map[string]interface{}{
		"entries":     hist.Entries,
		"formatted":   hist.Formatted,
		"tokens_used": hist.TokensUsed,
	}
	resultJSON, _ := json.Marshal(out)
	logToolQuery("recall_memory", args, len(resultJSON), 0, hist.TokensUsed, context.SavingsMeta{TokensUsed: hist.TokensUsed}, start, cpuStart, projectPath, "")
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
		Confirm:     boolArg(toolArgs, "confirm"),
	}
	// all=true is scoped only by an explicit project_path, never the server's default
	// project, so a bare all=true still needs confirm=true.
	if in.All {
		in.ProjectPath = strArg(toolArgs, "project_path")
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
	if v, ok := toolArgs["action"].(string); ok && v == "restore" {
		ref := strArg(toolArgs, "ref")
		if ref == "" {
			out := map[string]string{"error": "restore action requires a ref parameter"}
			resultJSON, _ := json.Marshal(out)
			logToolQuery("forget_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, pp, "")
			return out
		}
		res, err := memory.Restore(memory.RestoreInput{Refs: []string{ref}})
		if err != nil {
			out := map[string]string{"error": err.Error()}
			resultJSON, _ := json.Marshal(out)
			logToolQuery("forget_memory", args, len(resultJSON), 0, 0, context.SavingsMeta{}, start, cpuStart, pp, err.Error())
			return out
		}
		tokensUsed := 0
		if res.RestoredRefs != nil {
			tokensUsed = len(res.RestoredRefs)
		}
		out := map[string]interface{}{
			"restored_refs": res.RestoredRefs,
			"conflicts":     res.Conflicts,
		}
		resultJSON, _ := json.Marshal(out)
		logToolQuery("forget_memory", args, len(resultJSON), tokensUsed, 0, context.SavingsMeta{TokensUsed: tokensUsed}, start, cpuStart, pp, "")
		return out
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
