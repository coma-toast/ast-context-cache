package context

import (
	"encoding/json"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/render"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

var Emb embedder.Interface

// Capsule candidate limits: default when the caller omits limit, and the clamp ceiling.
const (
	defaultCapsuleLimit = 30
	maxCapsuleLimit     = 100
)

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
	limit := defaultCapsuleLimit
	if l, ok := args["limit"].(float64); ok && l > 0 {
		limit = min(int(l), maxCapsuleLimit)
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
	q := CandidateQuery{Stage: stage, Query: query, ProjectPath: projectPath, Limit: limit, Filters: filters}
	scored, pipeMetrics, cacheHit := RankedHybrid(q, Emb)
	scored = scored[:min(limit, len(scored))]
	entry := SearchTrailEntry("get_context_capsule", q, mode, scored, len(scored))
	p, err := packRanked(scored, query, projectPath, mode, sessionID, tokenBudget, ParsePrecisionArgs(args), false)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": err.Error()})
		return getContextResult{JSON: string(data)}
	}
	p.Savings.Mode = mode
	p.Savings.CacheHit = cacheHit
	resp := map[string]interface{}{
		"query":   query,
		"mode":    mode,
		"results": p.Results,
		"pipeline": map[string]interface{}{
			"bm25_candidates":   pipeMetrics.BM25Candidates,
			"vector_candidates": pipeMetrics.VectorCandidates,
			"hybrid_after_fuse": pipeMetrics.HybridAfterFuse,
		},
	}
	p.ApplyTo(resp)
	if tokenBudget > 0 {
		resp["token_budget"] = tokenBudget
		resp["tokens_remaining"] = tokenBudget - p.Savings.TokensUsed
	}
	codescripts.AttachHints(resp, "get_context_capsule", query, projectPath, p.Results)
	finalData, _ := json.Marshal(resp)
	return getContextResult{JSON: string(finalData), Savings: p.Savings, CacheHit: cacheHit, Trail: entry}
}

// PackScoredResults formats hybrid/vector search hits (used by search_semantic) after the
// precision pass for query (vector-only weak check). The entry carries the pre-dedup hit count
// and hits for the search trail; the caller fills in the tool, query, filters and doc type.
func PackScoredResults(scored []search.ScoredResult, limit int, projectPath, query, mode, sessionID string, tokenBudget int, args PrecisionArgs) (Packed, trail.Entry, error) {
	if mode == "" {
		mode = "skeleton"
	}
	scored = scored[:min(limit, len(scored))]
	entry := SearchTrailEntry("", CandidateQuery{ProjectPath: projectPath}, mode, scored, len(scored))
	p, err := packRanked(scored, query, projectPath, mode, sessionID, tokenBudget, args, true)
	p.Savings.Mode = mode
	return p, entry, err
}

// Packed is a code search's packed response: the results, their savings, and what the
// precision pass and token budget held back.
type Packed struct {
	Results   []map[string]interface{}
	Savings   SavingsMeta
	Withheld  Withheld
	NoMatch   *render.NoMatch
	Collapsed []render.Collapse
}

// ApplyTo writes the savings, withheld, no_match and collapsed fields into resp.
func (p Packed) ApplyTo(resp map[string]interface{}) {
	p.Savings.ApplyTo(resp)
	p.Withheld.ApplyTo(resp)
	if p.NoMatch != nil {
		resp["no_match"] = p.NoMatch
	}
	if len(p.Collapsed) > 0 {
		resp["collapsed"] = p.Collapsed
	}
}

// packRanked runs the precision pass over scored and packs what it keeps: the top hit's edit
// view for mode=edit under feature_mode_v2 (full for every hit with the flag off), the token
// budgeted hits otherwise.
func packRanked(scored []search.ScoredResult, query, projectPath, mode, sessionID string, tokenBudget int, args PrecisionArgs, semantic bool) (Packed, error) {
	prec := ApplyPrecision(scored, query, projectPath, args, semantic)
	fileCache := map[string][]string{}
	var p Packed
	switch {
	case mode == "edit" && flags.Enabled(flags.KeyModeV2):
		if len(prec.Kept) > 0 {
			hit := hitFromScored(prec.Kept[0], projectPath)
			var err error
			if p, err = PackEdit(hit.Data, projectPath, sessionID, tokenBudget, fileCache); err != nil {
				return Packed{}, err
			}
			p.Withheld.Add(EstimateWithheld(prec.Kept[1:], projectPath, fileCache))
		}
	case mode == "edit":
		p = packHits(prec.Kept, projectPath, "full", sessionID, tokenBudget, fileCache)
	default:
		p = packHits(prec.Kept, projectPath, mode, sessionID, tokenBudget, fileCache)
	}
	p.Withheld.Add(EstimateWithheld(prec.Withheld, projectPath, fileCache))
	p.NoMatch, p.Collapsed = prec.NoMatch, prec.Collapsed
	return p, nil
}

// packMode resolves the mode a hit is delivered in: by delivered rank (EffectiveModeV2) with
// feature_mode_v2 on, by score ratio (EffectiveMode) otherwise.
func packMode(mode string, v2 bool, rank int, score, maxScore float64, fullCount int) string {
	if v2 {
		return EffectiveModeV2(mode, rank)
	}
	return EffectiveMode(mode, score, maxScore, fullCount)
}

// packHits packs scored in rank order within tokenBudget. Hits the session already has in a
// mode at least as rich are skipped (dedup); hits over the budget are skipped and withheld,
// and packing stops after five consecutive budget misses, withholding the rest.
func packHits(scored []search.ScoredResult, projectPath, mode, sessionID string, tokenBudget int, fileCache map[string][]string) Packed {
	v2 := flags.Enabled(flags.KeyModeV2)
	returned := ReturnedModes(sessionID)
	var p Packed
	var delivered []ReturnedSymbol
	var spans []LineSpan
	matchedFiles := map[string]bool{}
	fullCount, misses := 0, 0
	maxScore := 0.0
	if len(scored) > 0 {
		maxScore = scored[0].Score
	}
	for i, r := range scored {
		if misses >= maxBudgetMisses {
			p.Withheld.Add(EstimateWithheld(scored[i:], projectPath, fileCache))
			break
		}
		hit := hitFromScored(r, projectPath)
		data := hit.Data
		file, name := mapStr(data, "file"), mapStr(data, "name")
		startLine, endLine := hit.StartLine, hit.EndLine
		owner := projectlinks.OwningProject(file, projectPath)
		effective := packMode(mode, v2, len(p.Results), hit.Score, maxScore, fullCount)
		key := SymbolDedupKey(file, name, startLine)
		if DedupCovers(returned, key, effective) {
			p.Savings.DedupedCount++
			p.Savings.DedupTokensSaved += WouldSendTokens(file, name, owner, effective, startLine, endLine, 0, 0, 0, fileCache)
			continue
		}
		ApplyMode(data, effective, file, name, owner, startLine, endLine, fileCache)
		search.StripFusionKeys(data)
		data["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && p.Savings.TokensUsed+resultTokens > tokenBudget {
			misses++
			p.Withheld.Add(Withheld{Count: 1, Tokens: resultTokens})
			continue
		}
		misses = 0
		if effective == "full" {
			fullCount++
		}
		p.Savings.SymbolBaseline += FullSourceTokens(file, name, owner, startLine, endLine, fileCache)
		spans = append(spans, LineSpan{File: file, Start: startLine, End: endLine})
		p.Savings.TokensUsed += resultTokens
		matchedFiles[file] = true
		p.Results = append(p.Results, data)
		mergeReturnedMode(returned, key, effective)
		delivered = append(delivered, ReturnedSymbol{File: file, Name: name, ProjectPath: projectPath, StartLine: startLine, Mode: effective, Tokens: resultTokens})
	}
	MarkReturned(sessionID, delivered...)
	p.Savings.finish(matchedFiles, spans, fileCache)
	return p
}

// PackEdit packs the edit view of target (a hit or symbol map with an absolute file and line
// span): the target's exact source, then its callees' skeletons within tokenBudget (the
// target is always sent). Every delivered symbol is marked returned in the mode it was sent.
func PackEdit(target map[string]interface{}, projectPath, sessionID string, tokenBudget int, fileCache map[string][]string) (Packed, error) {
	view, callees, err := EditView(projectlinks.OwningProject(mapStr(target, "file"), projectPath), target, fileCache)
	if err != nil {
		return Packed{}, err
	}
	search.StripFusionKeys(view)
	var p Packed
	var delivered []ReturnedSymbol
	var spans []LineSpan
	matchedFiles := map[string]bool{}
	for i, sym := range append([]map[string]interface{}{view}, callees...) {
		file, name := mapStr(sym, "file"), mapStr(sym, "name")
		startLine, endLine := coerceInt(sym["start_line"]), coerceInt(sym["end_line"])
		sym["file"] = db.RelPath(file, projectPath)
		resultJSON, _ := json.Marshal(sym)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if i > 0 && tokenBudget > 0 && p.Savings.TokensUsed+resultTokens > tokenBudget {
			p.Withheld.Add(Withheld{Count: 1, Tokens: resultTokens})
			continue
		}
		p.Savings.SymbolBaseline += FullSourceTokens(file, name, projectlinks.OwningProject(file, projectPath), startLine, endLine, fileCache)
		spans = append(spans, LineSpan{File: file, Start: startLine, End: endLine})
		p.Savings.TokensUsed += resultTokens
		matchedFiles[file] = true
		p.Results = append(p.Results, sym)
		delivered = append(delivered, ReturnedSymbol{File: file, Name: name, ProjectPath: projectPath, StartLine: startLine, Mode: mapStr(sym, "mode"), Tokens: resultTokens})
	}
	MarkReturned(sessionID, delivered...)
	p.Savings.finish(matchedFiles, spans, fileCache)
	return p, nil
}
