package context

import (
	"encoding/json"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

var Emb *embedder.Embedder

func HandleGetContext(args map[string]interface{}, projectPath string) string {
	query, _ := args["query"].(string)
	mode, _ := args["mode"].(string)
	sessionID, _ := args["session_id"].(string)
	tokenBudget := 0
	if tb, ok := args["token_budget"].(float64); ok && tb > 0 {
		tokenBudget = int(tb)
	}
	if mode == "" {
		mode = "full"
	}
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	if query == "" {
		return `{"error": "query required"}`
	}

	scored := search.HybridSearch(query, projectPath, Emb, 30)

	returnedFiles := GetReturnedFiles(sessionID)

	limit := 30
	if len(scored) < limit {
		limit = len(scored)
	}

	fileCache := map[string][]string{}
	var results []map[string]interface{}
	skipped := 0
	tokensUsed := 0
	for i := 0; i < limit; i++ {
		data := scored[i].Data
		file, _ := data["file"].(string)
		name, _ := data["name"].(string)
		startLine, _ := data["start_line"].(int)
		endLine, _ := data["end_line"].(int)

		if returnedFiles != nil && returnedFiles[file] {
			data["_deduped"] = true
			results = append(results, data)
			skipped++
			continue
		}

		switch mode {
		case "skeleton":
			if startLine > 0 && endLine > 0 {
				var skeleton string
				db.DB.QueryRow("SELECT COALESCE(skeleton,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
					file, name, projectPath, startLine).Scan(&skeleton)
				if skeleton != "" {
					data["skeleton"] = skeleton
				} else if src := indexer.ReadSourceRange(file, startLine, endLine, fileCache); src != "" {
					lang := indexer.GetLanguage(file)
					kind, _ := data["kind"].(string)
					data["skeleton"] = indexer.ExtractSkeleton(src, lang, kind)
				}
			}
		case "summary":
			var summary string
			if name != "" {
				db.DB.QueryRow("SELECT summary_text FROM summaries WHERE file_path = ? AND symbol_name = ? AND project_path = ?",
					file, name, projectPath).Scan(&summary)
			}
			if summary == "" {
				db.DB.QueryRow("SELECT summary_text FROM summaries WHERE file_path = ? AND (symbol_name IS NULL OR symbol_name = '') AND project_path = ?",
					file, projectPath).Scan(&summary)
			}
			if summary != "" {
				data["summary"] = summary
			} else {
				var skeleton string
				db.DB.QueryRow("SELECT COALESCE(skeleton,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
					file, name, projectPath, startLine).Scan(&skeleton)
				if skeleton != "" {
					data["skeleton"] = skeleton
					data["_fallback"] = "skeleton"
				}
			}
		default: // "full"
			if startLine > 0 && endLine > 0 {
				if src := indexer.ReadSourceRange(file, startLine, endLine, fileCache); src != "" {
					data["source"] = src
				}
			}
		}

		resultJSON, _ := json.Marshal(data)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && tokensUsed+resultTokens > tokenBudget {
			break
		}
		tokensUsed += resultTokens

		results = append(results, data)
		LogReturned(sessionID, file, 0, mode, resultTokens)
	}

	resp := map[string]interface{}{"query": query, "mode": mode, "results": results, "tokens_used": tokensUsed}
	if tokenBudget > 0 {
		resp["token_budget"] = tokenBudget
		resp["tokens_remaining"] = tokenBudget - tokensUsed
	}
	if skipped > 0 {
		resp["deduped"] = skipped
	}
	finalData, _ := json.Marshal(resp)
	return string(finalData)
}
