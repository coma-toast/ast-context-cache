package context

import (
	"encoding/json"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

var Emb embedder.Interface

func HandleGetContext(args map[string]interface{}, projectPath string) string {
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
		return `{"error": "project_path required"}`
	}
	if query == "" {
		return `{"error": "query required"}`
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
			return cached
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
	fullBaselineTokens := 0
	fullCount := 0
	maxScore := 0.0
	if len(scored) > 0 {
		maxScore = scored[0].Score
	}
	for i := 0; i < limit; i++ {
		data := scored[i].Data
		file, _ := data["file"].(string)
		name, _ := data["name"].(string)
		startLine, _ := data["start_line"].(int)
		endLine, _ := data["end_line"].(int)
		if startLine == 0 {
			db.DB.QueryRow("SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE file = ? AND name = ? AND project_path = ? LIMIT 1",
				file, name, projectPath).Scan(&startLine, &endLine)
		}
		if returnedSymbols != nil && returnedSymbols[SymbolDedupKey(file, name, startLine)] {
			skipped++
			continue
		}
		effectiveMode := EffectiveMode(mode, scored[i].Score, maxScore, fullCount)
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
		if effectiveMode == "full" && startLine > 0 && endLine > 0 {
			src := indexer.ReadSourceRange(file, startLine, endLine, fileCache)
			fullBaselineTokens += db.EstimateTokens(src)
		}
		results = append(results, data)
		LogReturned(sessionID, file, name, projectPath, startLine, mode, resultTokens)
	}
	fileBaselineTokens := 0
	for f := range matchedFiles {
		if lines, ok := fileCache[f]; ok {
			fileBaselineTokens += db.EstimateTokens(strings.Join(lines, "\n"))
		}
	}
	resp := map[string]interface{}{
		"query":                query,
		"mode":                 mode,
		"results":              results,
		"tokens_used":          tokensUsed,
		"file_baseline_tokens": fileBaselineTokens,
		"full_baseline_tokens": fullBaselineTokens,
		"pipeline": map[string]interface{}{
			"bm25_candidates":   pipeMetrics.BM25Candidates,
			"vector_candidates": pipeMetrics.VectorCandidates,
			"hybrid_after_fuse": pipeMetrics.HybridAfterFuse,
		},
	}
	if tokenBudget > 0 {
		resp["token_budget"] = tokenBudget
		resp["tokens_remaining"] = tokenBudget - tokensUsed
	}
	if skipped > 0 {
		resp["deduped"] = skipped
	}
	finalData, _ := json.Marshal(resp)
	resultStr := string(finalData)
	if useQueryCache {
		cache.GlobalCache.Set(cacheKey, resultStr)
	}
	return resultStr
}

// PackScoredResults formats hybrid/vector search hits (used by search_semantic).
func PackScoredResults(scored []search.ScoredResult, limit int, projectPath, mode, sessionID string, tokenBudget int) (results []map[string]interface{}, tokensUsed, skipped, fullBaseline int) {
	if mode == "" {
		mode = "skeleton"
	}
	returnedSymbols := GetReturnedSymbolKeys(sessionID)
	if len(scored) < limit {
		limit = len(scored)
	}
	fileCache := map[string][]string{}
	fullCount := 0
	maxScore := 0.0
	if len(scored) > 0 {
		maxScore = scored[0].Score
	}
	for i := 0; i < limit; i++ {
		data := scored[i].Data
		file, _ := data["file"].(string)
		name, _ := data["name"].(string)
		startLine, _ := data["start_line"].(int)
		endLine, _ := data["end_line"].(int)
		if startLine == 0 {
			db.DB.QueryRow("SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE file = ? AND name = ? AND project_path = ? LIMIT 1",
				file, name, projectPath).Scan(&startLine, &endLine)
			data["start_line"] = startLine
			data["end_line"] = endLine
		}
		if returnedSymbols != nil && returnedSymbols[SymbolDedupKey(file, name, startLine)] {
			skipped++
			continue
		}
		effectiveMode := EffectiveMode(mode, scored[i].Score, maxScore, fullCount)
		ApplyMode(data, effectiveMode, file, name, projectPath, startLine, endLine, fileCache)
		if effectiveMode == "full" {
			fullCount++
			if src, ok := data["source"].(string); ok {
				fullBaseline += db.EstimateTokens(src)
			}
		}
		data["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && tokensUsed+resultTokens > tokenBudget {
			break
		}
		tokensUsed += resultTokens
		results = append(results, data)
		LogReturned(sessionID, file, name, projectPath, startLine, mode, resultTokens)
	}
	return results, tokensUsed, skipped, fullBaseline
}
