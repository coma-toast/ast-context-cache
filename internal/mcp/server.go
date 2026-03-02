package mcp

import (
	"encoding/json"
	"net/http"
	"os"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/impact"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

var emb *embedder.Embedder

func SetEmbedder(e *embedder.Embedder) {
	emb = e
}

func GetEmbedder() *embedder.Embedder {
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

	var result interface{}
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
		const baselineContextTokens = 4000
		tokensSaved := baselineContextTokens - outputTokens
		if tokensSaved < 0 {
			tokensSaved = 0
		}

		db.LogQuery(toolName, args, len(contextStr), inputTokens, outputTokens, tokensSaved, float64(time.Since(start).Milliseconds()), projectPath, "")

		resultWithMeta := map[string]interface{}{
			"query":         query,
			"results":       json.RawMessage(contextStr),
			"tokens_saved":  tokensSaved,
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		resultJSON, _ := json.Marshal(resultWithMeta)
		result = json.RawMessage(resultJSON)
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
				results := make([]map[string]interface{}, len(scored))
				for i, s := range scored {
					results[i] = s.Data
					file, _ := s.Data["file"].(string)
					if file != "" {
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
								}
							}
							rows.Close()
						}
					}
				}
				data, _ := json.Marshal(map[string]interface{}{
					"query":         query,
					"results":       results,
					"total_vectors": search.Cache.Count(projectPath),
				})
				result = json.RawMessage(data)
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
			result = json.RawMessage(handleFileContext(file, projectPath, mode))
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
	default:
		result = map[string]string{"error": "not implemented: " + toolName}
	}

	if toolName != "get_context_capsule" {
		resultJSON, _ := json.Marshal(result)
		db.LogQuery(toolName, args, len(resultJSON), 0, 0, 0, float64(time.Since(start).Milliseconds()), projectPath, "")
	}
	json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      rpcReq.ID,
		Result:  result,
	})
}
