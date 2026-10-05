package context

import (
	"encoding/json"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

var Emb embedder.Interface

type getContextResult struct {
	JSON     string
	Savings  SavingsMeta
	CacheHit bool
	// Trail describes the search for the session's search trail; empty when the call failed.
	Trail trail.Entry
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
	stage := "capsule:bm25"
	if Emb != nil {
		embedqueue.EnsureProjectEmbeddings(projectPath)
		stage = "capsule:hybrid"
	}
	q := CandidateQuery{Stage: stage, Query: query, ProjectPath: projectPath, Limit: 30, Filters: filters}
	scored, pipeMetrics, cacheHit := RankedHybrid(q, Emb)
	returned := ReturnedKeys(sessionID)
	var delivered []ReturnedSymbol
	if len(scored) < limit {
		limit = len(scored)
	}
	entry := SearchTrailEntry("get_context_capsule", q, mode, scored, limit)
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
		owner := projectlinks.OwningProject(file, projectPath)
		key := SymbolDedupKey(file, name, startLine)
		if _, dup := returned[key]; dup {
			skipped++
			dedupTokens += WouldSendTokens(file, name, owner, mode, startLine, endLine, hit.Score, maxScore, fullCount, fileCache)
			continue
		}
		effectiveMode := EffectiveMode(mode, hit.Score, maxScore, fullCount)
		ApplyMode(data, effectiveMode, file, name, owner, startLine, endLine, fileCache)
		if effectiveMode == "full" {
			fullCount++
		}
		data["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && tokensUsed+resultTokens > tokenBudget {
			break
		}
		symbolBaseline += FullSourceTokens(file, name, owner, startLine, endLine, fileCache)
		tokensUsed += resultTokens
		matchedFiles[file] = true
		results = append(results, data)
		returned[key] = struct{}{}
		delivered = append(delivered, ReturnedSymbol{File: file, Name: name, ProjectPath: projectPath, StartLine: startLine, Mode: mode, Tokens: resultTokens})
	}
	MarkReturned(sessionID, delivered...)
	fileBaseline := FileBaselineTokens(matchedFiles, fileCache)
	savings := ComputeSavings(tokensUsed, symbolBaseline, fileBaseline, dedupTokens)
	savings.DedupedCount = skipped
	savings.Mode = mode
	savings.CacheHit = cacheHit
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
	codescripts.AttachHints(resp, "get_context_capsule", query, projectPath, results)
	finalData, _ := json.Marshal(resp)
	return getContextResult{JSON: string(finalData), Savings: savings, CacheHit: cacheHit, Trail: entry}
}

// PackScoredResults formats hybrid/vector search hits (used by search_semantic). entry carries
// the pre-dedup hit count and hits for the search trail; the caller fills in the tool, query,
// filters and doc type.
func PackScoredResults(scored []search.ScoredResult, limit int, projectPath, mode, sessionID string, tokenBudget int) (results []map[string]interface{}, savings SavingsMeta, entry trail.Entry) {
	if mode == "" {
		mode = "skeleton"
	}
	savings.Mode = mode
	returned := ReturnedKeys(sessionID)
	var delivered []ReturnedSymbol
	if len(scored) < limit {
		limit = len(scored)
	}
	entry = SearchTrailEntry("", CandidateQuery{ProjectPath: projectPath}, mode, scored, limit)
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
		owner := projectlinks.OwningProject(file, projectPath)
		key := SymbolDedupKey(file, name, startLine)
		if _, dup := returned[key]; dup {
			savings.DedupedCount++
			savings.DedupTokensSaved += WouldSendTokens(file, name, owner, mode, startLine, endLine, hit.Score, maxScore, fullCount, fileCache)
			continue
		}
		effectiveMode := EffectiveMode(mode, hit.Score, maxScore, fullCount)
		ApplyMode(data, effectiveMode, file, name, owner, startLine, endLine, fileCache)
		if effectiveMode == "full" {
			fullCount++
		}
		data["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && savings.TokensUsed+resultTokens > tokenBudget {
			break
		}
		savings.SymbolBaseline += FullSourceTokens(file, name, owner, startLine, endLine, fileCache)
		savings.TokensUsed += resultTokens
		matchedFiles[file] = true
		results = append(results, data)
		returned[key] = struct{}{}
		delivered = append(delivered, ReturnedSymbol{File: file, Name: name, ProjectPath: projectPath, StartLine: startLine, Mode: mode, Tokens: resultTokens})
	}
	MarkReturned(sessionID, delivered...)
	savings.FileBaseline = FileBaselineTokens(matchedFiles, fileCache)
	computed := ComputeSavings(savings.TokensUsed, savings.SymbolBaseline, savings.FileBaseline, savings.DedupTokensSaved)
	savings.TokensSaved = computed.TokensSaved
	savings.SavingsVsFiles = computed.SavingsVsFiles
	return results, savings, entry
}
