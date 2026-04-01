package mcp

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
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

	var chunks []RetrieveChunk
	var codeResults, docResults int
	totalTokens := 0

	codeChunks, codeCount := retrieveCode(query, projectPath, limit, includeSource)
	chunks = append(chunks, codeChunks...)
	codeResults = codeCount

	if includeDocs {
		docChunks, docCount := retrieveDocs(query, limit/2)
		chunks = append(chunks, docChunks...)
		docResults = docCount
	}

	chunks = rankAndDedup(chunks)
	chunks, totalTokens = budgetChunks(chunks, tokenBudget)

	result := RetrieveResult{
		Query:  query,
		Chunks: chunks,
		Stats: RetrieveStats{
			CodeResults: codeResults,
			DocResults:  docResults,
			TotalTokens: totalTokens,
		},
	}

	result.Context = assembleContext(chunks, format)

	resultJSON, _ := json.Marshal(result)
	return map[string]interface{}{
		"result": json.RawMessage(resultJSON),
	}
}

func retrieveCode(query, projectPath string, limit int, includeSource bool) ([]RetrieveChunk, int) {
	results := search.HybridSearch(query, projectPath, emb, limit*2)
	if len(results) > limit {
		results = results[:limit]
	}

	var chunks []RetrieveChunk
	for _, r := range results {
		name, _ := r.Data["name"].(string)
		kind, _ := r.Data["kind"].(string)
		file, _ := r.Data["file"].(string)

		if file == "" || name == "" {
			continue
		}

		var code, skeleton string
		var startLine, endLine int
		db.DB.QueryRow(
			"SELECT COALESCE(code,''), COALESCE(skeleton,''), COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE name = ? AND file = ? AND project_path = ? LIMIT 1",
			name, file, projectPath).Scan(&code, &skeleton, &startLine, &endLine)

		content := ""
		if startLine > 0 && endLine > 0 {
			fileCache := map[string][]string{}
			content = indexer.ReadSourceRange(file, startLine, endLine, fileCache)
		}
		if content == "" && code != "" {
			content = code
		}
		if content == "" && skeleton != "" {
			content = skeleton
		}

		if content == "" {
			continue
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
	}

	return chunks, len(results)
}

func retrieveDocs(query string, limit int) ([]RetrieveChunk, int) {
	entries, err := docs.SearchDocs(query, limit)
	if err != nil {
		return nil, 0
	}

	var chunks []RetrieveChunk
	for _, e := range entries {
		chunks = append(chunks, RetrieveChunk{
			Type:    "doc",
			Name:    e.Title,
			Kind:    "documentation",
			File:    e.Path,
			Score:   0.5,
			Source:  "docs",
			Content: e.Content,
		})
	}

	return chunks, len(entries)
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
