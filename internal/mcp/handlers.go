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

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/dop251/goja"
)

func handleProjectMap(projectPath string, depth int) string {
	type fileEntry struct {
		path    string
		symbols []map[string]string
	}

	rows, err := db.DB.Query(
		"SELECT file, name, kind FROM symbols WHERE project_path = ? ORDER BY file, start_line",
		projectPath)
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

func handleFileContext(file, projectPath, mode string) string {
	rows, err := db.DB.Query(
		"SELECT name, kind, start_line, end_line, COALESCE(skeleton,''), COALESCE(code,'') FROM symbols WHERE file = ? AND project_path = ? ORDER BY start_line",
		file, projectPath)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": err.Error()})
		return string(data)
	}
	defer rows.Close()

	fileCache := map[string][]string{}
	var symbols []map[string]interface{}
	fullBaselineTokens := 0

	for rows.Next() {
		var name, kind, skeleton, code string
		var startLine, endLine int
		rows.Scan(&name, &kind, &startLine, &endLine, &skeleton, &code)

		sym := map[string]interface{}{
			"name":       name,
			"kind":       kind,
			"start_line": startLine,
			"end_line":   endLine,
		}

		var fullSrc string
		if startLine > 0 && endLine > 0 {
			fullSrc = indexer.ReadSourceRange(file, startLine, endLine, fileCache)
		}
		if fullSrc != "" {
			fullBaselineTokens += db.EstimateTokens(fullSrc)
		}

		switch mode {
		case "skeleton":
			if skeleton != "" {
				sym["skeleton"] = skeleton
			} else if fullSrc != "" {
				lang := indexer.GetLanguage(file)
				sym["skeleton"] = indexer.ExtractSkeleton(fullSrc, lang, kind)
			}
		case "summary":
			var summary string
			db.DB.QueryRow("SELECT summary_text FROM summaries WHERE file_path = ? AND symbol_name = ? AND project_path = ?",
				file, name, projectPath).Scan(&summary)
			if summary != "" {
				sym["summary"] = summary
			} else if skeleton != "" {
				sym["skeleton"] = skeleton
				sym["_fallback"] = "skeleton"
			}
		default: // "full"
			if fullSrc != "" {
				sym["source"] = fullSrc
			}
		}

		symbols = append(symbols, sym)
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

	data, _ := json.Marshal(map[string]interface{}{
		"file":                 db.RelPath(file, projectPath),
		"language":             lang,
		"mode":                 mode,
		"symbols":              symbols,
		"total":                len(symbols),
		"file_baseline_tokens": fileBaselineTokens,
		"full_baseline_tokens": fullBaselineTokens,
	})
	return string(data)
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

	if kind == "" || kind == "function" {
		rows, err = db.DB.Query(`
			SELECT s.name, s.file, s.kind 
			FROM symbols s
			WHERE s.project_path = ? AND s.kind IN ('function', 'method')
			AND NOT EXISTS (
				SELECT 1 FROM edges e 
				WHERE e.source_file = s.file AND e.source_symbol = s.name AND e.kind = 'call'
			)
			ORDER BY s.file, s.name
		`, projectPath)
	} else {
		rows, err = db.DB.Query(`
			SELECT s.name, s.file, s.kind 
			FROM symbols s
			WHERE s.project_path = ? AND s.kind = ?
			AND NOT EXISTS (
				SELECT 1 FROM edges e 
				WHERE e.source_file = s.file AND e.source_symbol = s.name AND e.kind = 'import'
			)
			ORDER BY s.file, s.name
		`, projectPath, kind)
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

	rows, err := db.DB.Query(`
		SELECT name, file, kind, complexity 
		FROM symbols 
		WHERE project_path = ? AND complexity >= ?
		ORDER BY complexity DESC
		LIMIT ?
	`, projectPath, threshold, limit)

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
	code, _ := args["code"].(string)
	dataStr, _ := args["data"].(string)
	timeoutSecs := 5
	if t, ok := args["timeout"].(float64); ok && t > 0 && t <= 30 {
		timeoutSecs = int(t)
	}

	if code == "" || dataStr == "" {
		return map[string]interface{}{"error": "code and data are required"}
	}

	var data interface{}
	if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
		return map[string]interface{}{"error": "invalid JSON in data: " + err.Error()}
	}

	vm := goja.New()
	vm.SetFieldNameMapper(goja.TagFieldNameMapper("json", true))

	vm.Set("DATA", data)
	vm.Set("console", map[string]interface{}{
		"log": func(args ...interface{}) {
		},
	})

	done := make(chan error, 1)
	var result goja.Value
	go func() {
		var err error
		result, err = vm.RunString(code)
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil {
			return map[string]interface{}{
				"error":      "execution error: " + err.Error(),
				"data_count": getDataCount(data),
			}
		}

		if result != nil {
			return map[string]interface{}{
				"result":     result.Export(),
				"data_count": getDataCount(data),
			}
		}
		return map[string]interface{}{
			"result":     nil,
			"data_count": getDataCount(data),
		}
	case <-func() chan struct{} {
		ch := make(chan struct{})
		go func() {
			time.Sleep(time.Duration(timeoutSecs) * time.Second)
			close(ch)
		}()
		return ch
	}():
		return map[string]interface{}{
			"error":      "timeout after " + strconv.Itoa(timeoutSecs) + " seconds",
			"data_count": getDataCount(data),
		}
	}
}

func getDataCount(data interface{}) int {
	if arr, ok := data.([]interface{}); ok {
		return len(arr)
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
