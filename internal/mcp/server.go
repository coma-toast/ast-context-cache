package mcp

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/impact"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

var (
	emb embedder.Interface
	// srvCfg is read through GetConfig; nil until SetConfig or the first GetConfig.
	srvCfg atomic.Pointer[ServerConfig]
)

// defaultDocSourcesPerPage matches the dashboard's DefaultDocSourcesPerPage
// (internal/dashboard/doc_sources_ui.go) — list_doc_sources used to return
// every tracked doc source unbounded (docs.ListSources() -> ListSourcesPaged
// with perPage=0), unlike the dashboard's own paginated view of the same
// table.
const defaultDocSourcesPerPage = 10

func SetEmbedder(e embedder.Interface) {
	emb = e
	docs.SetEmbedder(e)
	embedder.MarkReady()
}

func GetEmbedder() embedder.Interface {
	return emb
}

func EmbedderState() (state string, lastUse time.Duration) {
	return embedder.HealthState()
}

func EmbedderError() string {
	return embedder.HealthError()
}

func RecordEmbed() {
	embedder.MarkSuccess()
}

// SetConfig replaces the server config. It is stored atomically because MCP request
// goroutines and the dashboard read it concurrently with whoever sets it.
func SetConfig(cfg ServerConfig) {
	srvCfg.Store(&cfg)
}

// GetConfig returns the current server config, initializing it from the environment on
// first use.
func GetConfig() ServerConfig {
	if cfg := srvCfg.Load(); cfg != nil {
		return *cfg
	}
	cfg := DefaultConfig()
	srvCfg.CompareAndSwap(nil, &cfg)
	return *srvCfg.Load()
}

// NewHandler serves the Streamable HTTP /mcp endpoint for both protocol eras; see
// protocol.go for how a request's era is chosen.
func NewHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			handlePost(w, r)
		case http.MethodGet:
			handleGet(w, r)
		case http.MethodDelete:
			handleDelete(w, r)
		default:
			methodNotAllowed(w)
		}
	}
}

func handlePost(w http.ResponseWriter, r *http.Request) {
	var rpcReq JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&rpcReq); err != nil {
		writeRPC(w, http.StatusBadRequest, rpcError(nil, &JSONRPCError{Code: ParseError, Message: err.Error()}))
		return
	}
	// The spec never lets a notification or a client's response get a body back.
	if isNotification(rpcReq) {
		w.WriteHeader(http.StatusAccepted)
		return
	}
	if eraOf(r, rpcReq) == eraModern {
		handleModern(w, r, rpcReq)
		return
	}
	handleLegacy(w, r, rpcReq)
}

// handleLegacy serves the handshake era. Requests are answered statelessly even when their
// Mcp-Session-Id is unknown (say, after a restart): nothing a POST does depends on the
// session, so forcing a re-initialize would only interrupt the client.
func handleLegacy(w http.ResponseWriter, r *http.Request, rpcReq JSONRPCRequest) {
	if id := r.Header.Get(headerSessionID); id != "" {
		hub.touch(id)
	}
	w.Header().Set("Content-Type", "application/json")
	switch rpcReq.Method {
	case "initialize":
		clientVersion, _ := rpcReq.Params["protocolVersion"].(string)
		negotiated := negotiate(clientVersion)
		if negotiated >= sessionVersionMin {
			w.Header().Set(headerSessionID, hub.newSession(negotiated))
		}
		writeRPC(w, http.StatusOK, rpcResult(rpcReq.ID, initializeResult(negotiated)))
	case "ping":
		writeRPC(w, http.StatusOK, rpcResult(rpcReq.ID, map[string]any{}))
	default:
		if !dispatch(w, rpcReq) {
			writeRPC(w, http.StatusOK, rpcError(rpcReq.ID, &JSONRPCError{Code: MethodNotFound, Message: "Unknown method: " + rpcReq.Method}))
		}
	}
}

// handleModern serves the stateless 2026-07-28 era: validate the request headers against
// the body, then answer through the shared dispatch and add the fields this revision
// requires. Any Mcp-Session-Id header is ignored.
func handleModern(w http.ResponseWriter, r *http.Request, rpcReq JSONRPCRequest) {
	if rerr, status := validateModern(r, rpcReq); rerr != nil {
		writeRPC(w, status, rpcError(rpcReq.ID, rerr))
		return
	}
	switch rpcReq.Method {
	case "server/discover":
		writeRPC(w, http.StatusOK, rpcResult(rpcReq.ID, discoverResult()))
		return
	case "subscriptions/listen":
		serveListen(w, r, rpcReq)
		return
	}
	buf := newResponseBuffer()
	if !dispatch(buf, rpcReq) {
		writeRPC(w, http.StatusNotFound, rpcError(rpcReq.ID, &JSONRPCError{Code: MethodNotFound, Message: "Unknown method: " + rpcReq.Method}))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(buf.status)
	if _, err := w.Write(modernize(buf.body.Bytes(), rpcReq.Method)); err != nil {
		logger.Debug("Failed to write MCP response", "error", err)
	}
}

// dispatch answers the methods both eras share, reporting false for an unknown method.
func dispatch(w http.ResponseWriter, rpcReq JSONRPCRequest) bool {
	switch rpcReq.Method {
	case "tools/list":
		writeRPC(w, http.StatusOK, rpcResult(rpcReq.ID, map[string]any{"tools": FilterTools(GetConfig())}))
	case "prompts/list":
		writeRPC(w, http.StatusOK, rpcResult(rpcReq.ID, map[string]any{"prompts": GetPrompts()}))
	case "prompts/get":
		handlePromptGet(w, rpcReq)
	case "tools/call":
		handleToolCall(w, rpcReq)
	default:
		return false
	}
	return true
}

// handleGet opens a legacy SSE stream. The 2026-07-28 revision removed the GET endpoint,
// so modern clients get 405.
func handleGet(w http.ResponseWriter, r *http.Request) {
	if isModernHeader(r) {
		methodNotAllowed(w)
		return
	}
	if acceptsSSE(r) {
		serveLegacyStream(w, r)
		return
	}
	// Deprecated: a plain GET returning the tool list predates Streamable HTTP. It is kept
	// for scripts and older integrations that still read it; MCP clients use tools/list.
	writeRPC(w, http.StatusOK, rpcResult(1, map[string]any{"tools": FilterTools(GetConfig())}))
}

// handleDelete ends a legacy session. Modern clients have no sessions and get 405.
func handleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(headerSessionID)
	if isModernHeader(r) || id == "" {
		methodNotAllowed(w)
		return
	}
	if !hub.dropSession(id) {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// resultIsError reports whether a tool's already-marshaled JSON result carries a
// non-empty top-level "error" field — the convention every handler in this
// package uses for a failure. Every other failure besides tier/access denial
// used to report isError:false with the error buried in content[].text, making
// it indistinguishable from a real empty result to a caller that checks the
// protocol-level flag instead of parsing the body.
func resultIsError(resultJSON []byte) bool {
	var probe struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(resultJSON, &probe) != nil {
		return false
	}
	return probe.Error != ""
}

func handleToolCall(w http.ResponseWriter, rpcReq JSONRPCRequest) {
	start := time.Now()
	cpuStart := sys.SampleCPU()
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

	cfg := GetConfig()
	if ok, reason := toolAccessByName(toolName, cfg); !ok {
		mcpErr := map[string]interface{}{
			"content": []map[string]interface{}{
				{"type": "text", "text": ToolDenyMessage(toolName, cfg, reason)},
			},
			"isError": true,
		}
		json.NewEncoder(w).Encode(JSONRPCResponse{
			JSONRPC: JSONRPCVersion,
			ID:      rpcReq.ID,
			Result:  mcpErr,
		})
		return
	}

	projectPath := ""
	if toolArgs != nil {
		if pp, ok := toolArgs["project_path"].(string); ok {
			projectPath = pp
		}
	}
	if projectPath != "" {
		if abs, err := filepath.Abs(projectPath); err == nil {
			projectPath = watcher.NormalizeProjectPath(abs)
		}
	}

	if projectPath != "" {
		watcher.EnsureWatcher(projectPath)
	}
	sid := sessionArg(toolArgs)

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
			} else if info.IsDir() {
				// Directories run as a background job; see index_jobs.go.
				result = withLinkedProjects(handleIndexDirectory(path, projectPath), projectPath)
			} else {
				n, _, _, indexErr := indexer.IndexFile(path, projectPath)
				if indexErr != nil {
					result = map[string]string{"error": indexErr.Error()}
				} else {
					projectmeta.ClearDeleted(projectPath)
					embedqueue.UnmarkProjectCancelled(projectPath)
					if emb != nil {
						go embedqueue.SubmitPriority(path, projectPath, db.IsPinnedProject(projectPath))
					}
					result = withLinkedProjects(map[string]interface{}{"indexed": n}, projectPath)
				}
			}
		}
	case "index_status":
		result = indexStatusResult(projectPath)
	case "get_context_capsule":
		query := ""
		if q, ok := toolArgs["query"].(string); ok {
			query = q
		}
		ctxResult := context.HandleGetContextWithMeta(toolArgs, projectPath)
		inputTokens := db.EstimateTokens(query)
		outputTokens := db.EstimateTokens(ctxResult.JSON)
		logToolQuery(toolName, args, len(ctxResult.JSON), inputTokens, outputTokens, ctxResult.Savings, start, cpuStart, projectPath, "")
		recordSearch(sid, ctxResult.Trail)
		var parsed map[string]interface{}
		if err := json.Unmarshal([]byte(ctxResult.JSON), &parsed); err == nil {
			parsed["input_tokens"] = inputTokens
			parsed["output_tokens"] = outputTokens
			if ann := annotateSearch(sid, ctxResult.Trail, resultMaps(parsed["results"])); ann != nil {
				parsed["handoff"] = ann
			}
			resultJSON, _ := json.Marshal(parsed)
			result = json.RawMessage(resultJSON)
		} else {
			result = json.RawMessage(ctxResult.JSON)
		}
	case "get_impact_graph":
		sym := ""
		if s, ok := toolArgs["symbol"].(string); ok {
			sym = s
		}
		resultStr := impact.HandleImpactGraph(map[string]interface{}{"symbol": sym}, projectPath)
		result = json.RawMessage(resultStr)
	case "diff_impact":
		result = json.RawMessage(impact.HandleDiffImpact(toolArgs, projectPath))
	case "check_symbol_exists":
		result = json.RawMessage(impact.HandleCheckSymbolExists(toolArgs, projectPath))
	case "check_deletion_safety":
		result = json.RawMessage(impact.HandleCheckDeletionSafety(toolArgs, projectPath))
	case "cache_summary":
		result = handleCacheSummary(toolArgs, projectPath)
	case "search_semantic":
		query := ""
		if q, ok := toolArgs["query"].(string); ok {
			query = q
		}
		docType := ""
		if dt, ok := toolArgs["doc_type"].(string); ok {
			docType = dt
		}
		if query == "" {
			result = map[string]string{"error": "query is required"}
		} else if docType == "doc" {
			limit := 10
			if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
				limit = int(l)
			}
			r, err := docs.SearchDocsHybridResult(query, limit, emb)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				results := make([]map[string]interface{}, 0, len(r.Docs))
				for _, s := range r.Docs {
					results = append(results, map[string]interface{}{
						"title":     s.Entry.Title,
						"content":   s.Entry.Content,
						"path":      s.Entry.Path,
						"source_id": s.Entry.SourceID,
						"score":     s.Score,
						"doc_type":  "doc",
					})
				}
				result = withDocNoMatch(map[string]interface{}{
					"query":    query,
					"doc_type": "doc",
					"results":  results,
					"total":    len(results),
				}, r.BelowFloor)
			}
		} else if projectPath == "" {
			result = map[string]string{"error": "project_path required unless doc_type=doc"}
		} else if emb == nil {
			result = map[string]string{"error": "embedder not available"}
		} else {
			embedqueue.EnsureProjectEmbeddings(projectPath)
			limit := 10
			if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
				limit = int(l)
			}
			filters := search.ParseSearchFilters(toolArgs)
			scored, cacheHit, embErr := semanticCandidates(query, projectPath, docType, limit, filters)
			if embErr != nil {
				result = map[string]string{"error": "embed query: " + embErr.Error()}
			} else {
				mode := "skeleton"
				if m, ok := toolArgs["mode"].(string); ok && m != "" {
					mode = m
				}
				sessionID, _ := toolArgs["session_id"].(string)
				tokenBudget := 4000
				if tb, ok := toolArgs["token_budget"].(float64); ok && tb > 0 {
					tokenBudget = int(tb)
				}
				results, packSavings, hits := context.PackScoredResults(scored, limit, projectPath, mode, sessionID, tokenBudget)
				packSavings.CacheHit = cacheHit
				resp := map[string]interface{}{
					"query":         query,
					"mode":          mode,
					"results":       results,
					"total_vectors": search.Cache.Count(projectPath),
				}
				packSavings.ApplyTo(resp)
				if tokenBudget > 0 {
					resp["token_budget"] = tokenBudget
					resp["tokens_remaining"] = tokenBudget - packSavings.TokensUsed
				}
				codescripts.AttachHints(resp, "search_semantic", query, projectPath, results)
				entry := semanticTrail(hits, query, docType, projectPath, filters)
				recordSearch(sessionID, entry)
				if ann := annotateSearch(sessionID, entry, results); ann != nil {
					resp["handoff"] = ann
				}
				respData, _ := json.Marshal(resp)
				outTokens := db.EstimateTokens(string(respData))
				logToolQuery(toolName, args, len(respData), db.EstimateTokens(query), outTokens, packSavings, start, cpuStart, projectPath, "")
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
		mode := "skeleton"
		if m, ok := toolArgs["mode"].(string); ok && m != "" {
			mode = m
		}
		sessionID, _ := toolArgs["session_id"].(string)
		tokenBudget := 0
		if tb, ok := toolArgs["token_budget"].(float64); ok && tb > 0 {
			tokenBudget = int(tb)
		}
		if file == "" || projectPath == "" {
			result = map[string]string{"error": "file and project_path required"}
		} else {
			fc := handleFileContextWithMeta(file, projectPath, mode, sessionID, tokenBudget)
			outTokens := db.EstimateTokens(fc.JSON)
			logToolQuery(toolName, args, len(fc.JSON), 0, outTokens, fc.Savings, start, cpuStart, projectPath, "")
			recordSearch(sessionID, fc.Trail)
			result = json.RawMessage(annotateFileContext(sessionID, fc.Trail, fc.JSON))
			loggedToolCall = true
		}
	case "analyze_dead_code":
		result = handleAnalyzeDeadCode(toolArgs, projectPath)
	case "analyze_complexity":
		result = handleAnalyzeComplexity(toolArgs, projectPath)
	case "execute_code":
		ec := handleExecuteCodeWithMeta(toolArgs)
		result = ec.Result
		resultJSON, _ := json.Marshal(ec.Result)
		logToolQuery(toolName, args, len(resultJSON), 0, ec.Savings.TokensUsed, ec.Savings, start, cpuStart, projectPath, "")
		loggedToolCall = true
	case "export_bundle":
		result = handleExportBundle(toolArgs)
	case "import_bundle":
		result = handleImportBundle(toolArgs)
	case "search_docs":
		query, _ := toolArgs["query"].(string)
		limit := 10
		if l, ok := toolArgs["limit"].(float64); ok && l > 0 {
			limit = int(l)
		}
		if query == "" {
			result = map[string]string{"error": "query is required"}
		} else {
			result = handleSearchDocs(query, limit)
		}
	case "fetch_doc":
		name, _ := toolArgs["name"].(string)
		docType, _ := toolArgs["type"].(string)
		docURL, _ := toolArgs["url"].(string)
		version, _ := toolArgs["version"].(string)
		force, _ := toolArgs["force_refresh"].(bool)
		renderJS, _ := toolArgs["render_js"].(bool)
		if name == "" || docType == "" || docURL == "" {
			result = map[string]string{"error": "name, type, and url are required"}
		} else {
			wantRender := renderJS || strings.EqualFold(docType, "webpage")
			storedType := docs.NormalizeDocType(docType, renderJS)
			id, entries, refreshed, usedPlaywright, err := docs.FetchAndCache(name, docType, docURL, version, force, renderJS)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				result = map[string]interface{}{
					"id":              id,
					"name":            name,
					"url":             docURL,
					"type":            storedType,
					"rendered":        usedPlaywright,
					"render_fallback": wantRender && !usedPlaywright,
					"cached":          true,
					"refreshed":       refreshed,
					"entries":         entries,
					"total":           len(entries),
				}
			}
		}
	case "add_doc_source":
		name, _ := toolArgs["name"].(string)
		docType, _ := toolArgs["type"].(string)
		docURL, _ := toolArgs["url"].(string)
		version, _ := toolArgs["version"].(string)
		if name == "" || docType == "" || docURL == "" {
			result = map[string]string{"error": "name, type, and url are required"}
		} else {
			id, err := docs.AddSource(name, docType, docURL, version)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				go func() { _, _ = docs.UpdateSource(id) }()
				result = map[string]interface{}{
					"status": "added",
					"id":     id,
					"name":   name,
					"url":    docURL,
				}
			}
		}
	case "remove_doc_source":
		idFloat, _ := toolArgs["id"].(float64)
		id := int(idFloat)
		if id == 0 {
			result = map[string]string{"error": "id is required"}
		} else {
			err := docs.RemoveSource(id)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				result = map[string]interface{}{"status": "removed", "id": id}
			}
		}
	case "list_doc_sources":
		page := 1
		if v, ok := toolArgs["page"].(float64); ok && v > 0 {
			page = int(v)
		}
		perPage := defaultDocSourcesPerPage
		if v, ok := toolArgs["per_page"].(float64); ok && v > 0 {
			perPage = int(v)
		}
		sources, total, actualPage, err := docs.ListSourcesPaged(page, perPage)
		if err != nil {
			result = map[string]string{"error": err.Error()}
		} else {
			result = map[string]interface{}{
				"sources":  sources,
				"total":    total,
				"page":     actualPage,
				"per_page": perPage,
			}
		}
	case "update_doc_source":
		idFloat, _ := toolArgs["id"].(float64)
		id := int(idFloat)
		if id == 0 {
			result = map[string]string{"error": "id is required"}
		} else {
			_, err := docs.UpdateSource(id)
			if err != nil {
				result = map[string]string{"error": err.Error()}
			} else {
				result = map[string]interface{}{"status": "updated", "id": id}
			}
		}
	case "retrieve":
		query, _ := toolArgs["query"].(string)
		if query == "" || projectPath == "" {
			result = map[string]string{"error": "query and project_path required"}
		} else {
			retrieveResult := HandleRetrieve(toolArgs, projectPath)
			if errMsg, ok := retrieveResult["error"].(string); ok {
				result = map[string]string{"error": errMsg}
			} else {
				result = retrieveResult["result"]
				var parsed map[string]interface{}
				if err := json.Unmarshal([]byte(result.(json.RawMessage)), &parsed); err == nil {
					ctxLen := len(parsed["context"].(string))
					outTokens := db.EstimateTokens(parsed["context"].(string))
					mode := "skeleton"
					if m, ok := toolArgs["mode"].(string); ok && m != "" {
						mode = m
					}
					savings := context.SavingsMeta{Mode: mode, TokensUsed: outTokens}
					if stats, ok := parsed["stats"].(map[string]interface{}); ok {
						if v, ok := stats["total_tokens"].(float64); ok {
							savings.TokensUsed = int(v)
						}
						if v, ok := stats["symbol_baseline_tokens"].(float64); ok {
							savings.SymbolBaseline = int(v)
						}
						if v, ok := stats["dedup_tokens_saved"].(float64); ok {
							savings.DedupTokensSaved = int(v)
						}
						if v, ok := stats["tokens_saved"].(float64); ok {
							savings.TokensSaved = int(v)
						}
						if v, ok := stats["deduped"].(float64); ok {
							savings.DedupedCount = int(v)
						}
					}
					if savings.TokensSaved == 0 {
						computed := context.ComputeSavings(savings.TokensUsed, savings.SymbolBaseline, 0, savings.DedupTokensSaved)
						savings.TokensSaved = computed.TokensSaved
					}
					logToolQuery(toolName, args, ctxLen, db.EstimateTokens(query), outTokens, savings, start, cpuStart, projectPath, "")
					loggedToolCall = true
				}
			}
		}
	default:
		if memResult, ok, _ := handleMemoryTool(toolName, toolArgs, args, emb, start, cpuStart, projectPath); ok {
			result = memResult
			loggedToolCall = true
		} else if ctxResult, ok, _ := handleContextTool(toolName, toolArgs, args, emb, start, cpuStart, projectPath); ok {
			result = ctxResult
			loggedToolCall = true
		} else {
			result = map[string]string{"error": "not implemented: " + toolName}
		}
	}

	if toolName != "get_context_capsule" && !loggedToolCall {
		resultJSON, _ := json.Marshal(result)
		logToolQuery(toolName, args, len(resultJSON), 0, db.EstimateTokens(string(resultJSON)), context.SavingsMeta{}, start, cpuStart, projectPath, "")
	}
	// MCP tools/call response must have result.content[].text and isError so clients pass tool output to the model.
	resultJSON, _ := json.Marshal(result)
	mcpResult := map[string]interface{}{
		"content": []map[string]interface{}{
			{"type": "text", "text": string(resultJSON)},
		},
		"isError": resultIsError(resultJSON),
	}
	json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      rpcReq.ID,
		Result:  mcpResult,
	})
}

func logToolQuery(toolName string, args map[string]interface{}, resultChars, inputTokens, outputTokens int, savings context.SavingsMeta, start time.Time, cpuStart sys.CPUSample, projectPath, errMsg string) {
	db.LogQuery(toolName, args, db.QueryLogMetrics{
		ResultChars:      resultChars,
		InputTokens:      inputTokens,
		OutputTokens:     outputTokens,
		TokensUsed:       savings.TokensUsed,
		TokensSaved:      savings.TokensSaved,
		SymbolBaseline:   savings.SymbolBaseline,
		FileBaseline:     savings.FileBaseline,
		DedupTokensSaved: savings.DedupTokensSaved,
		SavingsVsFiles:   savings.SavingsVsFiles,
		DedupedCount:     savings.DedupedCount,
		Mode:             savings.Mode,
		CacheHit:         savings.CacheHit,
		DurationMs:       float64(time.Since(start).Milliseconds()),
		CpuMs:            sys.DeltaMs(cpuStart, sys.SampleCPU()),
	}, projectPath, errMsg)
}
