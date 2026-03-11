package mcp

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/impact"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

var emb embedder.Interface

func SetEmbedder(e embedder.Interface) {
	emb = e
}

func GetEmbedder() embedder.Interface {
	return emb
}

func NewHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method == "GET" {
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      1,
				Result:  map[string]interface{}{"tools": GetTools()},
			})
			return
		}

		var rpcReq JSONRPCRequest
		if err := json.NewDecoder(r.Body).Decode(&rpcReq); err != nil {
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      nil,
				Error:   &JSONRPCError{Code: ParseError, Message: err.Error()},
			})
			return
		}

		switch rpcReq.Method {
		case "initialize":
			result := map[string]interface{}{
				"protocolVersion": "2024-11-05",
				"capabilities": map[string]interface{}{
					"tools": map[string]interface{}{},
				},
				"serverInfo": map[string]interface{}{
					"name":    "ast-context-cache",
					"version": "1.0.0",
				},
			}
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Result:  result,
			})
		case "initialized":
			return
		case "tools/list":
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Result:  map[string]interface{}{"tools": GetTools()},
			})
		case "prompts/list":
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Result:  map[string]interface{}{"prompts": GetPrompts()},
			})
		case "prompts/get":
			handlePromptGet(w, rpcReq)
		case "tools/call":
			handleToolCall(w, rpcReq)
		default:
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Error:   &JSONRPCError{Code: MethodNotFound, Message: "Unknown method: " + rpcReq.Method},
			})
		}
	}
}

func handleToolCall(w http.ResponseWriter, rpcReq JSONRPCRequest) {
	start := time.Now()
	args := rpcReq.Params
	if args == nil {
		args = make(map[string]any)
	}

	toolName := ""
	if name, ok := args["name"].(string); ok {
		toolName = name
	}

	var toolArgs map[string]interface{}
	if a, ok := args["arguments"].(map[string]interface{}); ok {
		toolArgs = a
	} else {
		toolArgs = args
	}

	projectPath := ""
	if toolArgs != nil {
		if pp, ok := toolArgs["project_path"].(string); ok {
			projectPath = pp
		}
	}
	if projectPath != "" {
		if abs, err := filepath.Abs(projectPath); err == nil {
			projectPath = filepath.Clean(abs)
		}
	}

	if projectPath != "" {
		watcher.EnsureWatcher(projectPath)
	}

	var result interface{}
	loggedToolCall := false
	switch toolName {
	case "index_files":
		path, _ := toolArgs["path"].(string)
		if path == "" || projectPath == "" {
			result = map[string]string{"error": "path and project_path required"}
		} else {
			info, statErr := os.Stat(path)
			if statErr != nil {
				result = map[string]string{"error": "path not found: " + statErr.Error()}
			} else {
				var n int
				var indexErr error
				if info.IsDir() {
					n, indexErr = indexer.IndexDirectory(path, projectPath)
					if indexErr == nil {
						go watcher.StartWatcher(projectPath)
						if emb != nil {
							go indexer.EmbedDirectorySymbols(emb, path, projectPath)
						}
					}
				} else {
					n, indexErr = indexer.IndexFile(path, projectPath)
					if indexErr == nil && emb != nil {
						go indexer.EmbedFileSymbols(emb, path, projectPath)
					}
				}
				if indexErr != nil {
					result = map[string]string{"error": indexErr.Error()}
				} else {
					result = map[string]int{"indexed": n}
				}
			}
		}
	case "index_status":
		result, _ = indexer.GetIndexStats(projectPath)
	case "get_context_capsule":
		query := ""
		if q, ok := toolArgs["query"].(string); ok {
			query = q
		}
		mode := ""
		if m, ok := toolArgs["mode"].(string); ok {
			mode = m
		}
		sessionID := ""
		if s, ok := toolArgs["session_id"].(string); ok {
			sessionID = s
		}
		var tokenBudget float64
		if tb, ok := toolArgs["token_budget"].(float64); ok {
			tokenBudget = tb
		}
		contextStr := context.HandleGetContext(map[string]interface{}{"query": query, "mode": mode, "session_id": sessionID, "token_budget": tokenBudget}, projectPath)

		inputTokens := db.EstimateTokens(query)
		outputTokens := db.EstimateTokens(contextStr)

		var parsed map[string]interface{}
		fileBaseline, fullBaseline := 0, 0
		if err := json.Unmarshal([]byte(contextStr), &parsed); err == nil {
			if v, ok := parsed["file_baseline_tokens"].(float64); ok {
				fileBaseline = int(v)
			}
			if v, ok := parsed["full_baseline_tokens"].(float64); ok {
				fullBaseline = int(v)
			}
		}

		tokensSaved := fullBaseline - outputTokens
		if tokensSaved < 0 {
			tokensSaved = 0
		}

		db.LogQuery(toolName, args, len(contextStr), inputTokens, outputTokens, tokensSaved, fileBaseline, fullBaseline, float64(time.Since(start).Milliseconds()), projectPath, "")

		if parsed != nil {
			parsed["tokens_saved"] = tokensSaved
			parsed["input_tokens"] = inputTokens
			parsed["output_tokens"] = outputTokens
			resultJSON, _ := json.Marshal(parsed)
			result = json.RawMessage(resultJSON)
		} else {
			result = json.RawMessage(contextStr)
		}
	case "get_impact_graph":
		sym := ""
		if s, ok := toolArgs["symbol"].(string); ok {
			sym = s
		}
		resultStr := impact.HandleImpactGraph(map[string]interface{}{"symbol": sym}, projectPath)
		result = json.RawMessage(resultStr)
	case "cache_summary":
		file, _ := toolArgs["file"].(string)
		summary, _ := toolArgs["summary"].(string)
		symbol, _ := toolArgs["symbol"].(string)
		if file == "" || summary == "" || projectPath == "" {
			result = map[string]string{"error": "file, summary, and project_path required"}
		} else {
			contentHash := search.ContentHash(summary)
			_, err := db.DB.Exec(
				`INSERT INTO summaries (file_path, symbol_name, summary_text, content_hash, project_path) 
				 VALUES (?, ?, ?, ?, ?)
				 ON CONFLICT(file_path, symbol_name, project_path) DO UPDATE SET summary_text=excluded.summary_text, content_hash=excluded.content_hash, created_at=datetime('now')`,
				file, symbol, summary, contentHash, projectPath)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				result = map[string]string{"status": "cached", "file": file, "symbol": symbol}
			}
		}
	case "search_semantic":
		query := ""
		if q, ok := toolArgs["query"].(string); ok {
			query = q
		}
		if query == "" || projectPath == "" {
			result = map[string]string{"error": "query and project_path required"}
		} else if emb == nil {
			result = map[string]string{"error": "embedder not available"}
		} else {
			limit := 10
			if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
				limit = int(l)
			}
			docType := ""
			if dt, ok := toolArgs["doc_type"].(string); ok {
				docType = dt
			}
			queryVec, embErr := emb.EmbedSingle(query)
			if embErr != nil {
				result = map[string]string{"error": "embed query: " + embErr.Error()}
			} else {
				scored := search.Cache.Search(queryVec, projectPath, docType, limit)
				fileCache := map[string][]string{}
				matchedFiles := map[string]bool{}
				fullBaselineTokens := 0
				results := make([]map[string]interface{}, len(scored))
				for i, s := range scored {
					results[i] = s.Data
					file, _ := s.Data["file"].(string)
					if file != "" {
						matchedFiles[file] = true
						rows, qErr := db.DB.Query("SELECT start_line, end_line FROM symbols WHERE file = ? AND name = ? AND project_path = ? LIMIT 1",
							file, s.Data["name"], projectPath)
						if qErr == nil {
							if rows.Next() {
								var sl, el int
								rows.Scan(&sl, &el)
								results[i]["start_line"] = sl
								results[i]["end_line"] = el
								if src := indexer.ReadSourceRange(file, sl, el, fileCache); src != "" {
									results[i]["source"] = src
									fullBaselineTokens += db.EstimateTokens(src)
								}
							}
							rows.Close()
						}
					}
					results[i]["file"] = db.RelPath(file, projectPath)
				}
				fileBaselineTokens := 0
				for f := range matchedFiles {
					if lines, ok := fileCache[f]; ok {
						fileBaselineTokens += db.EstimateTokens(strings.Join(lines, "\n"))
					}
				}
				respData, _ := json.Marshal(map[string]interface{}{
					"query":                query,
					"results":              results,
					"total_vectors":        search.Cache.Count(projectPath),
					"file_baseline_tokens": fileBaselineTokens,
					"full_baseline_tokens": fullBaselineTokens,
				})
				outTokens := db.EstimateTokens(string(respData))
				saved := fullBaselineTokens - outTokens
				if saved < 0 {
					saved = 0
				}
				db.LogQuery(toolName, args, len(respData), db.EstimateTokens(query), outTokens, saved, fileBaselineTokens, fullBaselineTokens, float64(time.Since(start).Milliseconds()), projectPath, "")
				loggedToolCall = true
				result = json.RawMessage(respData)
			}
		}
	case "get_project_map":
		if projectPath == "" {
			result = map[string]string{"error": "project_path required"}
		} else {
			depth := 2
			if d, ok := toolArgs["depth"].(float64); ok && d >= 1 && d <= 3 {
				depth = int(d)
			}
			result = json.RawMessage(handleProjectMap(projectPath, depth))
		}
	case "get_file_context":
		file, _ := toolArgs["file"].(string)
		mode := "full"
		if m, ok := toolArgs["mode"].(string); ok && m != "" {
			mode = m
		}
		if file == "" || projectPath == "" {
			result = map[string]string{"error": "file and project_path required"}
		} else {
			fcStr := handleFileContext(file, projectPath, mode)
			result = json.RawMessage(fcStr)
			var fcParsed map[string]interface{}
			if err := json.Unmarshal([]byte(fcStr), &fcParsed); err == nil {
				fileBase, fullBase, outTokens := 0, 0, db.EstimateTokens(fcStr)
				if v, ok := fcParsed["file_baseline_tokens"].(float64); ok {
					fileBase = int(v)
				}
				if v, ok := fcParsed["full_baseline_tokens"].(float64); ok {
					fullBase = int(v)
				}
				saved := fullBase - outTokens
				if saved < 0 {
					saved = 0
				}
				db.LogQuery(toolName, args, len(fcStr), 0, outTokens, saved, fileBase, fullBase, float64(time.Since(start).Milliseconds()), projectPath, "")
				loggedToolCall = true
			}
		}
	case "sync_remote":
		result = json.RawMessage(handleSyncRemote(toolArgs, projectPath))
	case "reset_project":
		pp := ""
		if p, ok := toolArgs["project_path"].(string); ok {
			pp = p
		}
		if pp == "" {
			result = map[string]string{"error": "project_path required"}
		} else {
			_, err := db.DB.Exec("DELETE FROM symbols WHERE project_path = ?", pp)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				db.DB.Exec("DELETE FROM edges WHERE project_path = ?", pp)
				result = map[string]string{"status": "deleted", "project_path": pp}
			}
		}
	case "reset_all":
		_, err := db.DB.Exec("DELETE FROM symbols")
		if err != nil {
			result = map[string]string{"error": err.Error()}
		} else {
			db.DB.Exec("DELETE FROM edges")
			result = map[string]string{"status": "deleted", "message": "All indexed data cleared"}
		}
	case "analyze_dead_code":
		result = handleAnalyzeDeadCode(toolArgs, projectPath)
	case "analyze_complexity":
		result = handleAnalyzeComplexity(toolArgs, projectPath)
	case "execute_code":
		result = handleExecuteCode(toolArgs)
	case "export_bundle":
		result = handleExportBundle(toolArgs)
	case "import_bundle":
		result = handleImportBundle(toolArgs)
	default:
		result = map[string]string{"error": "not implemented: " + toolName}
	}

	if toolName != "get_context_capsule" && !loggedToolCall {
		resultJSON, _ := json.Marshal(result)
		db.LogQuery(toolName, args, len(resultJSON), 0, 0, 0, 0, 0, float64(time.Since(start).Milliseconds()), projectPath, "")
	}
	json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      rpcReq.ID,
		Result:  result,
	})
}
