package context

import (
	"encoding/json"
	"strings"

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
	tokenBudget := 4000
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
	matchedFiles := map[string]bool{}
	var results []map[string]interface{}
	skipped := 0
	tokensUsed := 0
	fullBaselineTokens := 0
	fullCount := 0
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

		matchedFiles[file] = true

		// Always read the full source (cached) to compute the full-mode baseline
		var fullSrc string
		if startLine > 0 && endLine > 0 {
			fullSrc = indexer.ReadSourceRange(file, startLine, endLine, fileCache)
		}
		if fullSrc != "" {
			fullBaselineTokens += db.EstimateTokens(fullSrc)
		}

		effectiveMode := mode
		if mode == "auto" {
			if fullCount < 3 {
				effectiveMode = "full"
			} else {
				effectiveMode = "skeleton"
			}
		}

		switch effectiveMode {
		case "skeleton":
			if startLine > 0 && endLine > 0 {
				var skeleton string
				db.DB.QueryRow("SELECT COALESCE(skeleton,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
					file, name, projectPath, startLine).Scan(&skeleton)
				if skeleton != "" {
					data["skeleton"] = skeleton
				} else if fullSrc != "" {
					lang := indexer.GetLanguage(file)
					kind, _ := data["kind"].(string)
					data["skeleton"] = indexer.ExtractSkeleton(fullSrc, lang, kind)
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
			if fullSrc != "" {
				data["source"] = fullSrc
			}
		}

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

		results = append(results, data)
		LogReturned(sessionID, file, 0, mode, resultTokens)
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
	}
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
