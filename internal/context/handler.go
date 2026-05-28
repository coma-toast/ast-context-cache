package context

import (
	"encoding/json"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

var Emb embedder.Interface

type getContextResult struct {
	JSON     string
	Savings  SavingsMeta
	CacheHit bool
}

func HandleGetContext(args map[string]interface{}, projectPath string) string {
	r := handleGetContext(args, projectPath)
	return r.JSON
}

func HandleGetContextWithMeta(args map[string]interface{}, projectPath string) getContextResult {
	return handleGetContext(args, projectPath)
}

func handleGetContext(args map[string]interface{}, projectPath string) getContextResult {
	query, _ := args["query"].(string)
	mode, _ := args["mode"].(string)
	sessionID, _ := args["session_id"].(string)
	limit := 30
	if l, ok := args["limit"].(float64); ok && l > 0 {
		limit = int(l)
	}
	tokenBudget := 4000
	if tb, ok := args["token_budget"].(float64); ok && tb > 0 {
		tokenBudget = int(tb)
	}
	if mode == "" {
		mode = "auto"
	}
	if projectPath == "" {
		data, _ := json.Marshal(map[string]string{"error": "project_path required"})
		return getContextResult{JSON: string(data)}
	}
	if query == "" {
		data, _ := json.Marshal(map[string]string{"error": "query required"})
		return getContextResult{JSON: string(data)}
	}
	filters := search.ParseSearchFilters(args)
	filtersKey := ""
	if filters != nil {
		filtersKey = filters.CacheKey()
	}
	useQueryCache := sessionID == ""
	cacheKey := cache.HashQuery(query, projectPath, mode, limit, filtersKey)
	if useQueryCache {
		if cached, found := cache.GlobalCache.Get(cacheKey); found {
			var parsed map[string]interface{}
			if json.Unmarshal([]byte(cached), &parsed) == nil {
				savings := ParseSavingsMeta(parsed, mode, true)
				savings.CacheHit = true
				return getContextResult{JSON: cached, Savings: savings, CacheHit: true}
			}
			return getContextResult{JSON: cached, CacheHit: true}
		}
	}
	scored, pipeMetrics := search.HybridSearch(query, projectPath, Emb, 30, filters)
	returnedSymbols := GetReturnedSymbolKeys(sessionID)
	if len(scored) < limit {
		limit = len(scored)
	}
	fileCache := map[string][]string{}
	matchedFiles := map[string]bool{}
	var results []map[string]interface{}
	skipped := 0
	tokensUsed := 0
	symbolBaseline := 0
	dedupTokens := 0
	fullCount := 0
	maxScore := 0.0
	if len(scored) > 0 {
		maxScore = scored[0].Score
	}
	for i := 0; i < limit; i++ {
		hit := hitFromScored(scored[i], projectPath)
		data := hit.Data
		file, _ := data["file"].(string)
		name, _ := data["name"].(string)
		startLine, endLine := hit.StartLine, hit.EndLine
		if returnedSymbols != nil && returnedSymbols[SymbolDedupKey(file, name, startLine)] {
			skipped++
			dedupTokens += WouldSendTokens(file, name, projectPath, mode, startLine, endLine, hit.Score, maxScore, fullCount, fileCache)
			continue
		}
		effectiveMode := EffectiveMode(mode, hit.Score, maxScore, fullCount)
		symbolBaseline += FullSourceTokens(file, name, projectPath, startLine, endLine, fileCache)
		ApplyMode(data, effectiveMode, file, name, projectPath, startLine, endLine, fileCache)
		if effectiveMode == "full" {
			fullCount++
		}
		data["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && tokensUsed+resultTokens > tokenBudget {
			break
		}
		tokensUsed += resultTokens
		matchedFiles[file] = true
		results = append(results, data)
		LogReturned(sessionID, file, name, projectPath, startLine, mode, resultTokens)
	}
	fileBaseline := FileBaselineTokens(matchedFiles, fileCache)
	savings := ComputeSavings(tokensUsed, symbolBaseline, fileBaseline, dedupTokens)
	savings.DedupedCount = skipped
	savings.Mode = mode
	resp := map[string]interface{}{
		"query":   query,
		"mode":    mode,
		"results": results,
		"pipeline": map[string]interface{}{
			"bm25_candidates":   pipeMetrics.BM25Candidates,
			"vector_candidates": pipeMetrics.VectorCandidates,
			"hybrid_after_fuse": pipeMetrics.HybridAfterFuse,
		},
	}
	savings.ApplyTo(resp)
	if tokenBudget > 0 {
		resp["token_budget"] = tokenBudget
		resp["tokens_remaining"] = tokenBudget - tokensUsed
	}
	finalData, _ := json.Marshal(resp)
	resultStr := string(finalData)
	if useQueryCache {
		cache.GlobalCache.Set(cacheKey, resultStr)
	}
	return getContextResult{JSON: resultStr, Savings: savings}
}

// PackScoredResults formats hybrid/vector search hits (used by search_semantic).
func PackScoredResults(scored []search.ScoredResult, limit int, projectPath, mode, sessionID string, tokenBudget int) (results []map[string]interface{}, savings SavingsMeta) {
	if mode == "" {
		mode = "skeleton"
	}
	savings.Mode = mode
	returnedSymbols := GetReturnedSymbolKeys(sessionID)
	if len(scored) < limit {
		limit = len(scored)
	}
	fileCache := map[string][]string{}
	matchedFiles := map[string]bool{}
	fullCount := 0
	maxScore := 0.0
	if len(scored) > 0 {
		maxScore = scored[0].Score
	}
	for i := 0; i < limit; i++ {
		hit := hitFromScored(scored[i], projectPath)
		data := hit.Data
		file, _ := data["file"].(string)
		name, _ := data["name"].(string)
		startLine, endLine := hit.StartLine, hit.EndLine
		if returnedSymbols != nil && returnedSymbols[SymbolDedupKey(file, name, startLine)] {
			savings.DedupedCount++
			savings.DedupTokensSaved += WouldSendTokens(file, name, projectPath, mode, startLine, endLine, hit.Score, maxScore, fullCount, fileCache)
			continue
		}
		effectiveMode := EffectiveMode(mode, hit.Score, maxScore, fullCount)
		savings.SymbolBaseline += FullSourceTokens(file, name, projectPath, startLine, endLine, fileCache)
		ApplyMode(data, effectiveMode, file, name, projectPath, startLine, endLine, fileCache)
		if effectiveMode == "full" {
			fullCount++
		}
		data["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && savings.TokensUsed+resultTokens > tokenBudget {
			break
		}
		savings.TokensUsed += resultTokens
		matchedFiles[file] = true
		results = append(results, data)
		LogReturned(sessionID, file, name, projectPath, startLine, mode, resultTokens)
	}
	savings.FileBaseline = FileBaselineTokens(matchedFiles, fileCache)
	computed := ComputeSavings(savings.TokensUsed, savings.SymbolBaseline, savings.FileBaseline, savings.DedupTokensSaved)
	savings.TokensSaved = computed.TokensSaved
	savings.SavingsVsFiles = computed.SavingsVsFiles
	return results, savings
}
