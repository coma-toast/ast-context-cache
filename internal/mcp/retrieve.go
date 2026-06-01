package mcp

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

type RetrieveResult struct {
	Query   string          `json:"query"`
	Context string          `json:"context"`
	Chunks  []RetrieveChunk `json:"chunks"`
	Stats   RetrieveStats   `json:"stats"`
}

type RetrieveChunk struct {
	Type    string  `json:"type"`
	Name    string  `json:"name"`
	Kind    string  `json:"kind"`
	File    string  `json:"file"`
	Score   float64 `json:"score"`
	Source  string  `json:"source"`
	Content string  `json:"content"`
}

type RetrieveStats struct {
	CodeResults  int     `json:"code_results"`
	DocResults   int     `json:"doc_results"`
	TotalTokens  int     `json:"total_tokens"`
	SearchTimeMs float64 `json:"search_time_ms"`
	// Pipeline observability (hybrid search + assembly)
	BM25Candidates     int     `json:"bm25_candidates,omitempty"`
	VectorCandidates   int     `json:"vector_candidates,omitempty"`
	HybridAfterFuse    int     `json:"hybrid_after_fuse,omitempty"`
	AfterDedup         int     `json:"after_dedup,omitempty"`
	ChunksInBudget     int     `json:"chunks_in_budget,omitempty"`
	TokensEstAllChunks int     `json:"tokens_est_all_chunks,omitempty"`
	CodeRetrieveMs     float64 `json:"code_retrieve_ms,omitempty"`
	DocsRetrieveMs     float64 `json:"docs_retrieve_ms,omitempty"`
	DedupBudgetMs        float64 `json:"dedup_budget_ms,omitempty"`
	SymbolBaselineTokens int     `json:"symbol_baseline_tokens,omitempty"`
	DedupTokensSaved     int     `json:"dedup_tokens_saved,omitempty"`
	DedupedCount         int     `json:"deduped_count,omitempty"`
	TokensSaved          int     `json:"tokens_saved,omitempty"`
	SavingsVsCandidates  int     `json:"savings_vs_candidates,omitempty"`
	BudgetTokensSaved    int     `json:"budget_tokens_saved,omitempty"`
}

type codeRetrieveMeta struct {
	dedupCount     int
	dedupTokens    int
	symbolBaseline int
}

func HandleRetrieve(args map[string]interface{}, projectPath string) map[string]interface{} {
	query, _ := args["query"].(string)
	if query == "" {
		return map[string]interface{}{"error": "query is required"}
	}

	limit := 10
	if l, ok := args["limit"].(float64); ok && l > 0 {
		limit = int(l)
	}

	tokenBudget := 4000
	if tb, ok := args["token_budget"].(float64); ok && tb > 0 {
		tokenBudget = int(tb)
	}

	includeDocs := true
	if id, ok := args["include_docs"].(bool); ok {
		includeDocs = id
	}

	includeSource := false
	if is, ok := args["include_source"].(bool); ok {
		includeSource = is
	}

	format := "markdown"
	if f, ok := args["format"].(string); ok {
		format = f
	}
	mode := "skeleton"
	if m, ok := args["mode"].(string); ok && m != "" {
		mode = m
	}
	sessionID, _ := args["session_id"].(string)

	filters := search.ParseSearchFilters(args)
	tStart := time.Now()

	tCode := time.Now()
	codeChunks, codeCount, hybridMetrics, codeMeta := retrieveCode(query, projectPath, limit, includeSource, mode, sessionID, filters)
	codeMs := float64(time.Since(tCode).Milliseconds())

	var docChunks []RetrieveChunk
	docCount := 0
	var docsMs float64
	if includeDocs {
		tDocs := time.Now()
		docChunks, docCount = retrieveDocs(query, limit/2)
		docsMs = float64(time.Since(tDocs).Milliseconds())
	}

	chunks := append(codeChunks, docChunks...)

	tDedup := time.Now()
	chunks = rankAndDedup(chunks)
	afterDedup := len(chunks)
	tokensEstAll := 0
	for _, c := range chunks {
		tokensEstAll += db.EstimateTokens(c.Content)
	}
	chunks, totalTokens := budgetChunks(chunks, tokenBudget)
	dedupBudgetMs := float64(time.Since(tDedup).Milliseconds())
	symbolBaseline := baselineForChunks(chunks, projectPath)
	budgetSaved := tokensEstAll - totalTokens
	if budgetSaved < 0 {
		budgetSaved = 0
	}
	savings := context.ComputeSavings(totalTokens, symbolBaseline, 0, codeMeta.dedupTokens)
	savings.DedupedCount = codeMeta.dedupCount
	savings.Mode = mode

	result := RetrieveResult{
		Query:  query,
		Chunks: chunks,
		Stats: RetrieveStats{
			CodeResults:          codeCount,
			DocResults:           docCount,
			TotalTokens:          totalTokens,
			SearchTimeMs:         float64(time.Since(tStart).Milliseconds()),
			BM25Candidates:       hybridMetrics.BM25Candidates,
			VectorCandidates:     hybridMetrics.VectorCandidates,
			HybridAfterFuse:      hybridMetrics.HybridAfterFuse,
			AfterDedup:           afterDedup,
			ChunksInBudget:       len(chunks),
			TokensEstAllChunks:   tokensEstAll,
			CodeRetrieveMs:       codeMs,
			DocsRetrieveMs:       docsMs,
			DedupBudgetMs:        dedupBudgetMs,
			SymbolBaselineTokens: symbolBaseline,
			DedupTokensSaved:     codeMeta.dedupTokens,
			DedupedCount:         codeMeta.dedupCount,
			TokensSaved:          savings.TokensSaved,
			SavingsVsCandidates:  budgetSaved,
			BudgetTokensSaved:    budgetSaved,
		},
	}

	result.Context = assembleContext(chunks, format)

	resultJSON, _ := json.Marshal(result)
	return map[string]interface{}{
		"result": json.RawMessage(resultJSON),
	}
}

func retrieveCode(query, projectPath string, limit int, includeSource bool, mode, sessionID string, filters *search.SearchFilters) ([]RetrieveChunk, int, *search.HybridSearchMetrics, codeRetrieveMeta) {
	results, metrics := search.HybridSearch(query, projectPath, emb, limit*2, filters)
	if len(results) > limit {
		results = results[:limit]
	}
	returnedSymbols := context.GetReturnedSymbolKeys(sessionID)
	fileCache := map[string][]string{}
	var chunks []RetrieveChunk
	meta := codeRetrieveMeta{}
	fullCount := 0
	maxScore := 0.0
	if len(results) > 0 {
		maxScore = results[0].Score
	}
	for _, r := range results {
		name, _ := r.Data["name"].(string)
		kind, _ := r.Data["kind"].(string)
		file, _ := r.Data["file"].(string)
		if file == "" || name == "" {
			continue
		}
		var startLine, endLine int
		db.DB.QueryRow(
			"SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE name = ? AND file = ? AND project_path = ? LIMIT 1",
			name, file, projectPath).Scan(&startLine, &endLine)
		if returnedSymbols != nil && returnedSymbols[context.SymbolDedupKey(file, name, startLine)] {
			meta.dedupCount++
			meta.dedupTokens += context.WouldSendTokens(file, name, projectPath, mode, startLine, endLine, r.Score, maxScore, fullCount, fileCache)
			continue
		}
		content := context.SymbolContentForRetrieve(file, name, projectPath, startLine, endLine, includeSource, mode, r.Score, maxScore, fullCount, fileCache)
		if content == "" {
			continue
		}
		if includeSource || mode == "full" || (mode == "auto" && context.EffectiveMode("auto", r.Score, maxScore, fullCount) == "full") {
			fullCount++
		}
		chunks = append(chunks, RetrieveChunk{
			Type:    "code",
			Name:    name,
			Kind:    kind,
			File:    db.RelPath(file, projectPath),
			Score:   r.Score,
			Source:  "code",
			Content: content,
		})
		context.LogReturned(sessionID, file, name, projectPath, startLine, mode, db.EstimateTokens(content))
	}
	return chunks, len(results), metrics, meta
}

func baselineForChunks(chunks []RetrieveChunk, projectPath string) int {
	fileCache := map[string][]string{}
	total := 0
	for _, c := range chunks {
		if c.Type == "doc" {
			total += db.EstimateTokens(c.Content)
			continue
		}
		absFile := c.File
		if projectPath != "" && !strings.HasPrefix(absFile, "/") {
			absFile = projectPath + "/" + c.File
		}
		var startLine, endLine int
		db.DB.QueryRow(
			"SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE name = ? AND file = ? AND project_path = ? LIMIT 1",
			c.Name, absFile, projectPath).Scan(&startLine, &endLine)
		total += context.FullSourceTokens(absFile, c.Name, projectPath, startLine, endLine, fileCache)
	}
	return total
}

func retrieveDocs(query string, limit int) ([]RetrieveChunk, int) {
	scored, err := docs.SearchDocsHybrid(query, limit, docs.Embedder())
	if err != nil || len(scored) == 0 {
		entries, err2 := docs.SearchDocs(query, limit)
		if err2 != nil {
			return nil, 0
		}
		scored = make([]docs.ScoredDoc, len(entries))
		for i, e := range entries {
			scored[i] = docs.ScoredDoc{Entry: e, Score: 0.5}
		}
	}
	var chunks []RetrieveChunk
	for _, s := range scored {
		score := s.Score
		if score <= 0 {
			score = 0.5
		}
		chunks = append(chunks, RetrieveChunk{
			Type:    "doc",
			Name:    s.Entry.Title,
			Kind:    "documentation",
			File:    s.Entry.Path,
			Score:   score,
			Source:  "docs",
			Content: s.Entry.Content,
		})
	}
	return chunks, len(chunks)
}

func rankAndDedup(chunks []RetrieveChunk) []RetrieveChunk {
	seen := map[string]bool{}
	var unique []RetrieveChunk

	for _, c := range chunks {
		key := c.Type + "|" + c.File + "|" + c.Name
		if seen[key] {
			continue
		}
		seen[key] = true
		unique = append(unique, c)
	}

	sort.Slice(unique, func(i, j int) bool {
		return unique[i].Score > unique[j].Score
	})

	return unique
}

func budgetChunks(chunks []RetrieveChunk, budget int) ([]RetrieveChunk, int) {
	var result []RetrieveChunk
	totalTokens := 0

	for _, c := range chunks {
		tokens := db.EstimateTokens(c.Content)
		if totalTokens+tokens > budget {
			break
		}
		totalTokens += tokens
		result = append(result, c)
	}

	return result, totalTokens
}

func assembleContext(chunks []RetrieveChunk, format string) string {
	if len(chunks) == 0 {
		return "No relevant context found."
	}

	var sb strings.Builder

	switch format {
	case "markdown":
		sb.WriteString("# Retrieved Context\n\n")
		for i, c := range chunks {
			sb.WriteString(fmt.Sprintf("## %d. %s (%s)\n\n", i+1, c.Name, c.Type))
			if c.Type == "code" {
				sb.WriteString(fmt.Sprintf("**File:** `%s` | **Kind:** `%s` | **Score:** %.3f\n\n", c.File, c.Kind, c.Score))
				sb.WriteString(fmt.Sprintf("```%s\n%s\n```\n\n", langFromExt(c.File), c.Content))
			} else {
				sb.WriteString(fmt.Sprintf("**Source:** Documentation | **Score:** %.3f\n\n", c.Score))
				sb.WriteString(c.Content + "\n\n")
			}
		}
	case "xml":
		sb.WriteString("<context>\n")
		for _, c := range chunks {
			sb.WriteString(fmt.Sprintf("  <chunk type=\"%s\" name=\"%s\" file=\"%s\" score=\"%.3f\">\n", c.Type, escapeXML(c.Name), escapeXML(c.File), c.Score))
			sb.WriteString(fmt.Sprintf("    <![CDATA[%s]]>\n", c.Content))
			sb.WriteString("  </chunk>\n")
		}
		sb.WriteString("</context>\n")
	case "json":
		data, _ := json.MarshalIndent(chunks, "", "  ")
		sb.Write(data)
	default:
		for _, c := range chunks {
			sb.WriteString(fmt.Sprintf("[%s] %s (%s)\n%s\n---\n", c.Type, c.Name, c.File, c.Content))
		}
	}

	return sb.String()
}

func langFromExt(file string) string {
	ext := strings.ToLower(file)
	if idx := strings.LastIndex(ext, "."); idx >= 0 {
		ext = ext[idx:]
	}
	switch ext {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".ts", ".tsx":
		return "typescript"
	case ".js", ".jsx":
		return "javascript"
	case ".rs":
		return "rust"
	case ".rb":
		return "ruby"
	case ".java":
		return "java"
	case ".sh":
		return "bash"
	case ".fish":
		return "fish"
	case ".yaml", ".yml":
		return "yaml"
	default:
		return ""
	}
}

func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	return s
}
