package mcp

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/dop251/goja"
)

func handleProjectMap(projectPath string, depth int) string {
	type fileEntry struct {
		path    string
		symbols []map[string]string
	}

	indexDB, err := indexDBOrErr()
	if err != nil {
		return indexDBErrJSON(err)
	}
	scopeFrag, scopeArgs := projectlinks.ScopeSQL("", projectPath)
	rows, err := indexDB.Query(
		"SELECT file, name, kind FROM symbols WHERE "+scopeFrag+" ORDER BY file, start_line",
		scopeArgs...)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": err.Error()})
		return string(data)
	}
	defer rows.Close()

	files := map[string][]map[string]string{}
	for rows.Next() {
		var file, name, kind string
		rows.Scan(&file, &name, &kind)
		files[file] = append(files[file], map[string]string{"name": name, "kind": kind})
	}

	sortedFiles := make([]string, 0, len(files))
	for f := range files {
		sortedFiles = append(sortedFiles, f)
	}
	sort.Strings(sortedFiles)

	type dirNode struct {
		Name  string                   `json:"name"`
		Type  string                   `json:"type"`
		Files []map[string]interface{} `json:"files,omitempty"`
		Dirs  []*dirNode               `json:"dirs,omitempty"`
	}

	dirs := map[string]*dirNode{}
	var topDirs []string

	for _, f := range sortedFiles {
		rel, err := filepath.Rel(projectPath, f)
		if err != nil {
			continue
		}
		dir := filepath.Dir(rel)
		if _, ok := dirs[dir]; !ok {
			dirs[dir] = &dirNode{Name: dir, Type: "directory"}
			topDirs = append(topDirs, dir)
		}

		if depth >= 2 {
			fileInfo := map[string]interface{}{
				"name":    filepath.Base(f),
				"symbols": len(files[f]),
			}
			if depth >= 3 {
				fileInfo["symbol_list"] = files[f]
			}
			dirs[dir].Files = append(dirs[dir].Files, fileInfo)
		}
	}

	sort.Strings(topDirs)

	result := map[string]interface{}{
		"depth":       depth,
		"total_files": len(sortedFiles),
	}

	if depth == 1 {
		dirNames := make([]string, len(topDirs))
		for i, d := range topDirs {
			dirNames[i] = d
		}
		result["directories"] = dirNames
	} else {
		tree := make([]map[string]interface{}, 0)
		for _, d := range topDirs {
			dn := dirs[d]
			entry := map[string]interface{}{
				"directory": dn.Name,
			}
			if dn.Files != nil {
				entry["files"] = dn.Files
			}
			tree = append(tree, entry)
		}
		result["tree"] = tree
	}

	data, _ := json.Marshal(result)
	return string(data)
}

type fileContextResult struct {
	JSON    string
	Savings context.SavingsMeta
}

func handleFileContext(file, projectPath, mode, sessionID string, tokenBudget int) string {
	return handleFileContextWithMeta(file, projectPath, mode, sessionID, tokenBudget).JSON
}

func handleFileContextWithMeta(file, projectPath, mode, sessionID string, tokenBudget int) fileContextResult {
	indexDB, err := indexDBOrErr()
	if err != nil {
		return fileContextResult{JSON: indexDBErrJSON(err)}
	}
	owner := projectlinks.OwningProject(file, projectPath)
	rows, err := indexDB.Query(
		"SELECT name, kind, start_line, end_line, COALESCE(skeleton,''), COALESCE(code,'') FROM symbols WHERE file = ? AND project_path = ? ORDER BY start_line",
		file, owner)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": err.Error()})
		return fileContextResult{JSON: string(data)}
	}
	defer rows.Close()
	returnedSymbols := context.GetReturnedSymbolKeys(sessionID)
	fileCache := map[string][]string{}
	var symbols []map[string]interface{}
	symbolBaseline := 0
	tokensUsed := 0
	dedupTokens := 0
	skipped := 0
	fullCount := 0
	maxScore := 1.0

	for rows.Next() {
		var name, kind, skeleton, code string
		var startLine, endLine int
		rows.Scan(&name, &kind, &startLine, &endLine, &skeleton, &code)
		if returnedSymbols != nil && returnedSymbols[context.SymbolDedupKey(file, name, startLine)] {
			skipped++
			dedupTokens += context.WouldSendTokens(file, name, projectPath, mode, startLine, endLine, maxScore, maxScore, fullCount, fileCache)
			continue
		}
		sym := map[string]interface{}{
			"name":       name,
			"kind":       kind,
			"start_line": startLine,
			"end_line":   endLine,
		}
		effectiveMode := context.EffectiveMode(mode, maxScore, maxScore, fullCount)
		context.ApplyMode(sym, effectiveMode, file, name, projectPath, startLine, endLine, fileCache)
		if effectiveMode == "full" {
			fullCount++
		}
		resultJSON, _ := json.Marshal(sym)
		resultTokens := db.EstimateTokens(string(resultJSON))
		if tokenBudget > 0 && tokensUsed+resultTokens > tokenBudget {
			break
		}
		symbolBaseline += context.FullSourceTokens(file, name, projectPath, startLine, endLine, fileCache)
		tokensUsed += resultTokens
		symbols = append(symbols, sym)
		context.LogReturned(sessionID, file, name, projectPath, startLine, mode, resultTokens)
	}

	lang := ""
	ext := strings.ToLower(filepath.Ext(file))
	switch ext {
	case ".py":
		lang = "Python"
	case ".go":
		lang = "Go"
	case ".js", ".jsx":
		lang = "JavaScript"
	case ".ts":
		lang = "TypeScript"
	case ".tsx":
		lang = "TSX"
	case ".sh":
		lang = "Bash"
	case ".fish":
		lang = "Fish"
	case ".yaml", ".yml":
		lang = "YAML"
	case ".tf", ".tfvars":
		lang = "HCL"
	}

	fileBaselineTokens := 0
	if lines, ok := fileCache[file]; ok {
		fileBaselineTokens = db.EstimateTokens(strings.Join(lines, "\n"))
	}
	savings := context.ComputeSavings(tokensUsed, symbolBaseline, fileBaselineTokens, dedupTokens)
	savings.DedupedCount = skipped
	savings.Mode = mode

	resp := map[string]interface{}{
		"file":     db.RelPath(file, projectPath),
		"language": lang,
		"mode":     mode,
		"symbols":  symbols,
		"total":    len(symbols),
	}
	savings.ApplyTo(resp)
	if tokenBudget > 0 {
		resp["token_budget"] = tokenBudget
		resp["tokens_remaining"] = tokenBudget - tokensUsed
	}
	data, _ := json.Marshal(resp)
	return fileContextResult{JSON: string(data), Savings: savings}
}

func handlePromptGet(w http.ResponseWriter, rpcReq JSONRPCRequest) {
	args := rpcReq.Params
	name := ""
	if n, ok := args["name"].(string); ok {
		name = n
	}

	prompts := GetPrompts()
	for _, p := range prompts {
		if p.Name == name {
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Result: map[string]interface{}{
					"prompt": map[string]string{
						"role":    "user",
						"content": p.Prompt,
					},
				},
			})
			return
		}
	}

	json.NewEncoder(w).Encode(JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      rpcReq.ID,
		Error:   &JSONRPCError{Code: InvalidParams, Message: "Prompt not found: " + name},
	})
}

func handleAnalyzeDeadCode(args map[string]interface{}, projectPath string) map[string]interface{} {
	kind := ""
	if k, ok := args["kind"].(string); ok {
		kind = k
	}

	var rows *sql.Rows
	var err error

	indexDB, err := indexDBOrErr()
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}

	if kind == "" || kind == "function" {
		scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
		rows, err = indexDB.Query(`
			SELECT s.name, s.file, s.kind 
			FROM symbols s
			WHERE `+scopeFrag+` AND s.kind IN ('function', 'method')
			AND NOT EXISTS (
				SELECT 1 FROM edges e 
				WHERE e.source_file = s.file AND e.source_symbol = s.name AND e.kind = 'call'
			)
			ORDER BY s.file, s.name
		`, scopeArgs...)
	} else {
		scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
		rows, err = indexDB.Query(`
			SELECT s.name, s.file, s.kind 
			FROM symbols s
			WHERE `+scopeFrag+` AND s.kind = ?
			AND NOT EXISTS (
				SELECT 1 FROM edges e 
				WHERE e.source_file = s.file AND e.source_symbol = s.name AND e.kind = 'import'
			)
			ORDER BY s.file, s.name
		`, append(scopeArgs, kind)...)
	}

	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var name, file, kind string
		rows.Scan(&name, &file, &kind)
		results = append(results, map[string]interface{}{
			"name": name,
			"file": db.RelPath(file, projectPath),
			"kind": kind,
		})
	}

	return map[string]interface{}{
		"results":     results,
		"total":       len(results),
		"description": "Symbols that are defined but never used",
	}
}

func handleAnalyzeComplexity(args map[string]interface{}, projectPath string) map[string]interface{} {
	threshold := 10
	if t, ok := args["threshold"].(float64); ok {
		threshold = int(t)
	}
	limit := 20
	if l, ok := args["limit"].(float64); ok {
		limit = int(l)
	}

	indexDB, err := indexDBOrErr()
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	scopeFrag, scopeArgs := projectlinks.ScopeSQL("", projectPath)
	rows, err := indexDB.Query(`
		SELECT name, file, kind, complexity 
		FROM symbols 
		WHERE `+scopeFrag+` AND complexity >= ?
		ORDER BY complexity DESC
		LIMIT ?
	`, append(scopeArgs, threshold, limit)...)

	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var name, file, kind string
		var complexity int
		rows.Scan(&name, &file, &kind, &complexity)
		results = append(results, map[string]interface{}{
			"name":       name,
			"file":       db.RelPath(file, projectPath),
			"kind":       kind,
			"complexity": complexity,
		})
	}

	return map[string]interface{}{
		"results":   results,
		"total":     len(results),
		"threshold": threshold,
	}
}

func handleExecuteCode(args map[string]interface{}) map[string]interface{} {
	return handleExecuteCodeWithMeta(args).Result
}

type executeCodeOutcome struct {
	Result  map[string]interface{}
	Savings context.SavingsMeta
}

func handleExecuteCodeWithMeta(args map[string]interface{}) executeCodeOutcome {
	code, _ := args["code"].(string)
	scriptID, _ := args["script_id"].(string)
	projectPath, _ := args["project_path"].(string)
	dataStr, _ := args["data"].(string)
	timeoutSecs := 5
	if t, ok := args["timeout"].(float64); ok && t > 0 && t <= 30 {
		timeoutSecs = int(t)
	}
	if dataStr == "" {
		return executeCodeOutcome{Result: map[string]interface{}{"error": "data is required"}}
	}
	if code == "" && scriptID != "" {
		resolved, err := codescripts.ResolveScript(scriptID, projectPath)
		if err != nil {
			return executeCodeOutcome{Result: map[string]interface{}{"error": err.Error()}}
		}
		code = resolved
	}
	if code == "" {
		return executeCodeOutcome{Result: map[string]interface{}{"error": "code or script_id is required"}}
	}
	dataBaseline := db.EstimateTokens(dataStr)
	var data interface{}
	if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
		return executeCodeOutcome{Result: map[string]interface{}{"error": "invalid JSON in data: " + err.Error()}}
	}
	vm := goja.New()
	vm.SetFieldNameMapper(goja.TagFieldNameMapper("json", true))
	vm.Set("DATA", data)
	vm.Set("console", map[string]interface{}{
		"log": func(args ...interface{}) {},
	})
	done := make(chan error, 1)
	var result goja.Value
	go func() {
		var err error
		wrapped := strings.TrimSpace(code)
		if !strings.HasPrefix(wrapped, "(function") {
			wrapped = "(function(){\n" + code + "\n})()"
		}
		result, err = vm.RunString(wrapped)
		done <- err
	}()
	fail := func(msg string) executeCodeOutcome {
		resp := map[string]interface{}{
			"error":                msg,
			"data_count":           getDataCount(data),
			"data_baseline_tokens": dataBaseline,
			"tokens_used":          0,
			"tokens_saved":         0,
		}
		if scriptID != "" {
			resp["script_id"] = scriptID
		}
		return executeCodeOutcome{Result: resp}
	}
	select {
	case err := <-done:
		if err != nil {
			return fail("execution error: " + err.Error())
		}
		var exported interface{}
		if result != nil {
			exported = result.Export()
		}
		outJSON, _ := json.Marshal(exported)
		outTokens := db.EstimateTokens(string(outJSON))
		saved := dataBaseline - outTokens
		if saved < 0 {
			saved = 0
		}
		mode := "code_script"
		if scriptID != "" {
			mode = scriptID
		}
		resp := map[string]interface{}{
			"result":               exported,
			"data_count":           getDataCount(data),
			"data_baseline_tokens": dataBaseline,
			"tokens_used":          outTokens,
			"tokens_saved":         saved,
		}
		if scriptID != "" {
			resp["script_id"] = scriptID
		}
		return executeCodeOutcome{
			Result: resp,
			Savings: context.SavingsMeta{
				Mode:           mode,
				SymbolBaseline: dataBaseline,
				TokensUsed:     outTokens,
				TokensSaved:    saved,
			},
		}
	case <-func() chan struct{} {
		ch := make(chan struct{})
		go func() {
			time.Sleep(time.Duration(timeoutSecs) * time.Second)
			close(ch)
		}()
		return ch
	}():
		return fail("timeout after " + strconv.Itoa(timeoutSecs) + " seconds")
	}
}

func getDataCount(data interface{}) int {
	if arr, ok := data.([]interface{}); ok {
		return len(arr)
	}
	if m, ok := data.(map[string]interface{}); ok {
		if arr, ok := m["results"].([]interface{}); ok {
			return len(arr)
		}
	}
	return 0
}

func handleExportBundle(args map[string]interface{}) map[string]interface{} {
	projectPath, _ := args["project_path"].(string)
	outputPath, _ := args["output_path"].(string)

	if projectPath == "" || outputPath == "" {
		return map[string]interface{}{"error": "project_path and output_path are required"}
	}

	return map[string]interface{}{
		"message": "Bundle export not yet implemented",
		"project": projectPath,
		"output":  outputPath,
	}
}

func handleImportBundle(args map[string]interface{}) map[string]interface{} {
	bundlePath, _ := args["bundle_path"].(string)

	if bundlePath == "" {
		return map[string]interface{}{"error": "bundle_path is required"}
	}

	return map[string]interface{}{
		"message":     "Bundle import not yet implemented",
		"bundle_path": bundlePath,
	}
}
