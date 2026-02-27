package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	_ "github.com/mattn/go-sqlite3"
	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/bash"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/python"
	"github.com/smacker/go-tree-sitter/typescript/tsx"
	"github.com/smacker/go-tree-sitter/typescript/typescript"
)

var (
	frontendDir string
)

const (
	MCP_PORT       = 7821
	DASHBOARD_PORT = 7830
	DB_PATH        = ".astcache/usage.db"
	FRONTEND_DIR   = "dist"
)

var dbPath string

func init() {
	home := os.Getenv("HOME")
	if home == "" {
		dbPath = DB_PATH
	} else {
		dbPath = home + "/.astcache/usage.db"
	}
}

var db *sql.DB

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type Stats struct {
	TotalQueries     int     `json:"total_queries"`
	TotalSessions    int     `json:"total_sessions"`
	TotalChars       int     `json:"total_chars"`
	TotalTokensSaved int     `json:"total_tokens_saved"`
	AvgDurationMs    float64 `json:"avg_duration_ms"`
	TodayQueries     int     `json:"today_queries"`
	TodayTokensSaved int     `json:"today_tokens_saved"`
}

type JSONRPCRequest struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      interface{}    `json:"id"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
}

const (
	JSONRPCVersion = "2.0"
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      interface{}   `json:"id"`
	Result  interface{}   `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

type ToolCallRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type ToolCallResponse struct {
	Content []ContentBlock `json:"content"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func initDB() error {
	os.MkdirAll(filepath.Dir(dbPath), 0755)
	var err error
	db, err = sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return err
	}

	db.Exec(`PRAGMA journal_mode=WAL`)
	db.Exec(`PRAGMA busy_timeout=5000`)

	db.Exec(`
		CREATE TABLE IF NOT EXISTS queries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			tool_name TEXT NOT NULL,
			arguments TEXT,
			result_chars INTEGER,
			input_tokens INTEGER DEFAULT 0,
			output_tokens INTEGER DEFAULT 0,
			tokens_saved INTEGER DEFAULT 0,
			duration_ms REAL,
			interface TEXT DEFAULT 'http',
			session_id TEXT,
			error TEXT,
			project_path TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_queries_project ON queries(project_path);
		CREATE TABLE IF NOT EXISTS symbols (
			id INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			kind TEXT NOT NULL,
			file TEXT NOT NULL,
			start_line INTEGER,
			end_line INTEGER,
			code TEXT,
			fqn TEXT,
			project_path TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
		CREATE TABLE IF NOT EXISTS edges (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_file TEXT NOT NULL,
			source_symbol TEXT,
			target TEXT NOT NULL,
			kind TEXT NOT NULL,
			project_path TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
		CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_file);
		CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
		CREATE INDEX IF NOT EXISTS idx_symbols_project ON symbols(project_path);
		CREATE INDEX IF NOT EXISTS idx_edges_project ON edges(project_path);
	`)

	db.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(name, fqn, code, content='symbols', content_rowid='id')`)

	ensureFTSTriggers()
	go db.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)

	return nil
}

func ensureFTSTriggers() {
	db.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_ins AFTER INSERT ON symbols BEGIN
		INSERT INTO symbols_fts(rowid, name, fqn, code) VALUES (new.id, new.name, new.fqn, new.code);
	END`)
	db.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_del AFTER DELETE ON symbols BEGIN
		INSERT INTO symbols_fts(symbols_fts, rowid, name, fqn, code) VALUES('delete', old.id, old.name, old.fqn, old.code);
	END`)
}

func logQuery(toolName string, args map[string]interface{}, resultChars int, inputTokens int, outputTokens int, tokensSaved int, durationMs float64, projectPath string, errMsg string) {
	sessionID := fmt.Sprintf("session-%d", time.Now().Unix()/3600)
	argsJSON, _ := json.Marshal(args)
	db.Exec("INSERT INTO queries (timestamp, tool_name, arguments, result_chars, input_tokens, output_tokens, tokens_saved, duration_ms, interface, session_id, error, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		time.Now().Format(time.RFC3339), toolName, string(argsJSON), resultChars, inputTokens, outputTokens, tokensSaved, durationMs, "http", sessionID, errMsg, projectPath)
}

func estimateTokens(text string) int {
	return len(text) / 4
}

func getLanguage(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".py":
		return "python"
	case ".js", ".jsx":
		return "javascript"
	case ".ts":
		return "typescript"
	case ".tsx":
		return "tsx"
	case ".go":
		return "go"
	case ".sh":
		return "bash"
	case ".fish":
		return "fish"
	}
	return ""
}

func isCodeFile(path string) bool {
	return getLanguage(path) != ""
}

func getSitterLanguage(lang string) *sitter.Language {
	switch lang {
	case "python":
		return python.GetLanguage()
	case "javascript":
		return javascript.GetLanguage()
	case "typescript":
		return typescript.GetLanguage()
	case "tsx":
		return tsx.GetLanguage()
	case "go":
		return golang.GetLanguage()
	case "bash":
		return bash.GetLanguage()
	}
	return nil
}

type symbolDef struct {
	name string
	kind string
}

func extractSymbol(node *sitter.Node, content []byte, lang string) *symbolDef {
	nodeType := node.Type()

	switch lang {
	case "python":
		switch nodeType {
		case "function_definition":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "class_definition":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "class"}
		case "decorated_definition":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				if s := extractSymbol(node.NamedChild(j), content, lang); s != nil {
					return s
				}
			}
		}

	case "javascript":
		switch nodeType {
		case "function_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "class_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "class"}
		case "lexical_declaration", "variable_declaration":
			if name := getVarDeclName(node, content); name != "" {
				return &symbolDef{name, "variable"}
			}
		case "export_statement":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				if s := extractSymbol(node.NamedChild(j), content, lang); s != nil {
					return s
				}
			}
		}

	case "typescript", "tsx":
		switch nodeType {
		case "function_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "class_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "class"}
		case "interface_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "interface"}
		case "type_alias_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "type"}
		case "enum_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "enum"}
		case "lexical_declaration", "variable_declaration":
			if name := getVarDeclName(node, content); name != "" {
				return &symbolDef{name, "variable"}
			}
		case "export_statement":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				if s := extractSymbol(node.NamedChild(j), content, lang); s != nil {
					return s
				}
			}
		}

	case "go":
		switch nodeType {
		case "function_declaration":
			return &symbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "method_declaration":
			return &symbolDef{getFirstChildByType(node, content, "field_identifier"), "method"}
		case "type_declaration":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				child := node.NamedChild(j)
				if child.Type() == "type_spec" {
					name := getFirstChildByType(child, content, "type_identifier")
					if name != "" {
						kind := "type"
						for k := 0; k < int(child.NamedChildCount()); k++ {
							switch child.NamedChild(k).Type() {
							case "struct_type":
								kind = "struct"
							case "interface_type":
								kind = "interface"
							}
						}
						return &symbolDef{name, kind}
					}
				}
			}
		}

	case "bash":
		if nodeType == "function_definition" {
			return &symbolDef{getFirstChildByType(node, content, "word"), "function"}
		}
	}

	return nil
}

func extractImports(node *sitter.Node, content []byte, lang string) []string {
	var imports []string
	nodeType := node.Type()

	switch lang {
	case "python":
		switch nodeType {
		case "import_statement":
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				if child.Type() == "dotted_name" || child.Type() == "aliased_import" {
					imports = append(imports, child.Content(content))
				}
			}
		case "import_from_statement":
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				if child.Type() == "dotted_name" || child.Type() == "relative_import" {
					imports = append(imports, child.Content(content))
					break
				}
			}
		}

	case "javascript", "typescript", "tsx":
		if nodeType == "import_statement" {
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				if child.Type() == "string" || child.Type() == "string_fragment" {
					src := strings.Trim(child.Content(content), "'\"")
					if src != "" {
						imports = append(imports, src)
					}
				}
			}
		}

	case "go":
		if nodeType == "import_declaration" {
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				switch child.Type() {
				case "import_spec":
					p := getFirstChildByType(child, content, "interpreted_string_literal")
					if p != "" {
						imports = append(imports, strings.Trim(p, "\""))
					}
				case "import_spec_list":
					for j := 0; j < int(child.NamedChildCount()); j++ {
						spec := child.NamedChild(j)
						if spec.Type() == "import_spec" {
							p := getFirstChildByType(spec, content, "interpreted_string_literal")
							if p != "" {
								imports = append(imports, strings.Trim(p, "\""))
							}
						}
					}
				}
			}
		}

	case "bash":
		if nodeType == "command" {
			name := getFirstChildByType(node, content, "command_name")
			if name == "source" || name == "." {
				for i := 0; i < int(node.NamedChildCount()); i++ {
					child := node.NamedChild(i)
					if child.Type() == "word" || child.Type() == "string" {
						val := strings.Trim(child.Content(content), "'\"")
						if val != name {
							imports = append(imports, val)
						}
					}
				}
			}
		}
	}

	return imports
}

func handleImpactGraph(args map[string]interface{}, projectPath string) string {
	symbol, _ := args["symbol"].(string)
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	if symbol == "" {
		return `{"error": "symbol required"}`
	}

	symbolLower := strings.ToLower(symbol)

	symbolRows, err := db.Query(
		"SELECT DISTINCT file FROM symbols WHERE project_path = ? AND LOWER(name) = ?",
		projectPath, symbolLower)
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, err.Error())
	}
	defer symbolRows.Close()

	symbolFiles := map[string]bool{}
	for symbolRows.Next() {
		var f string
		symbolRows.Scan(&f)
		symbolFiles[f] = true
	}

	type impactEntry struct {
		File   string `json:"file"`
		Target string `json:"target"`
		Kind   string `json:"kind"`
	}
	var impacts []impactEntry

	edgeRows, err := db.Query(
		"SELECT source_file, target, kind FROM edges WHERE project_path = ? AND (LOWER(target) LIKE ? OR LOWER(target) LIKE ?)",
		projectPath, "%"+symbolLower+"%", "%/"+symbolLower)
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, err.Error())
	}
	defer edgeRows.Close()

	seen := map[string]bool{}
	for edgeRows.Next() {
		var srcFile, target, kind string
		edgeRows.Scan(&srcFile, &target, &kind)
		if !seen[srcFile] {
			seen[srcFile] = true
			impacts = append(impacts, impactEntry{File: srcFile, Target: target, Kind: kind})
		}
	}

	for f := range symbolFiles {
		depRows, _ := db.Query(
			"SELECT source_file, target, kind FROM edges WHERE project_path = ? AND LOWER(target) LIKE ?",
			projectPath, "%"+strings.ToLower(filepath.Base(f))+"%")
		if depRows != nil {
			for depRows.Next() {
				var srcFile, target, kind string
				depRows.Scan(&srcFile, &target, &kind)
				if !seen[srcFile] {
					seen[srcFile] = true
					impacts = append(impacts, impactEntry{File: srcFile, Target: target, Kind: kind})
				}
			}
			depRows.Close()
		}
	}

	data, _ := json.Marshal(map[string]interface{}{
		"symbol":      symbol,
		"defined_in":  mapKeys(symbolFiles),
		"impacted_by": impacts,
		"total_files": len(seen),
	})
	return string(data)
}

func mapKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func indexFile(filePath, projectPath string) (int, error) {
	lang := getLanguage(filePath)
	if lang == "" {
		return 0, fmt.Errorf("unsupported: %s", filePath)
	}
	if lang == "fish" {
		return indexFishFile(filePath, projectPath)
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, err
	}

	sitterLang := getSitterLanguage(lang)
	if sitterLang == nil {
		return 0, fmt.Errorf("no parser for: %s", lang)
	}

	parser := sitter.NewParser()
	parser.SetLanguage(sitterLang)

	tree, err := parser.ParseCtx(context.Background(), nil, content)
	if err != nil {
		return 0, err
	}
	defer tree.Close()

	db.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath)
	db.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath)

	count := 0
	lines := strings.Split(string(content), "\n")
	root := tree.RootNode()
	for i := 0; i < int(root.NamedChildCount()); i++ {
		node := root.NamedChild(i)

		for _, imp := range extractImports(node, content, lang) {
			db.Exec("INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, ?, 'import', ?)",
				filePath, imp, projectPath)
		}

		sym := extractSymbol(node, content, lang)
		if sym == nil || sym.name == "" {
			continue
		}
		start := node.StartPoint()
		end := node.EndPoint()
		code := ""
		if int(start.Row) < len(lines) {
			code = strings.TrimSpace(lines[start.Row])
		}
		fqn := fmt.Sprintf("%s.%s", filepath.Base(filePath), sym.name)
		_, err := db.Exec("INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
			sym.name, sym.kind, filePath, start.Row+1, end.Row+1, code, fqn, projectPath)
		if err == nil {
			count++
		}
	}
	return count, nil
}

func getFirstChildByType(node *sitter.Node, content []byte, nodeType string) string {
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		if child.Type() == nodeType {
			return child.Content(content)
		}
	}
	return ""
}

func getVarDeclName(node *sitter.Node, content []byte) string {
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		if child.Type() == "variable_declarator" {
			return getFirstChildByType(child, content, "identifier")
		}
	}
	return ""
}

func indexFishFile(filePath, projectPath string) (int, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, err
	}

	db.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath)
	db.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath)

	lines := strings.Split(string(content), "\n")
	count := 0

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "source ") {
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				db.Exec("INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, ?, 'import', ?)",
					filePath, parts[1], projectPath)
			}
		}
	}

	type funcInfo struct {
		name  string
		line  int
		depth int
	}

	var funcStack []funcInfo
	depth := 0

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		if strings.HasPrefix(trimmed, "function ") {
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				funcStack = append(funcStack, funcInfo{name: parts[1], line: i, depth: depth})
			}
			depth++
			continue
		}

		if !strings.HasPrefix(trimmed, "else") {
			for _, kw := range []string{"if ", "for ", "while ", "switch "} {
				if strings.HasPrefix(trimmed, kw) {
					depth++
					break
				}
			}
			if trimmed == "begin" {
				depth++
			}
		}

		if trimmed == "end" || strings.HasPrefix(trimmed, "end;") || strings.HasPrefix(trimmed, "end #") {
			depth--
			if len(funcStack) > 0 && funcStack[len(funcStack)-1].depth == depth {
				fs := funcStack[len(funcStack)-1]
				funcStack = funcStack[:len(funcStack)-1]
				code := strings.TrimSpace(lines[fs.line])
				fqn := fmt.Sprintf("%s.%s", filepath.Base(filePath), fs.name)
				db.Exec("INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
					fs.name, "function", filePath, fs.line+1, i+1, code, fqn, projectPath)
				count++
			}
		}
	}
	return count, nil
}

func indexDirectory(dirPath, projectPath string) (int, error) {
	count := 0
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if shouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if !isCodeFile(path) {
			return nil
		}
		n, err := indexFile(path, projectPath)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		}
		count += n
		return nil
	})
	return count, err
}

func getIndexStats(projectPath string) (map[string]interface{}, error) {
	var nodes, files int
	var err error
	if projectPath != "" {
		err = db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", projectPath).Scan(&nodes, &files)
	} else {
		err = db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&nodes, &files)
	}
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"total_nodes": nodes, "total_files": files}, nil
}

func getTools() []Tool {
	return []Tool{
		{
			Name:        "get_context_capsule",
			Description: "Search indexed code symbols using BM25-ranked full-text search. Returns matching functions, classes, types with file paths, line ranges, and actual source code. Supports Python, JS/TS, Go, Bash, Fish. Eliminates the need to read entire files.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query":        map[string]string{"type": "string", "description": "Search query (function name, class name, type name, or keywords)"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
				},
				"required": []string{"query", "project_path"},
			},
		},
		{
			Name:        "index_files",
			Description: "Index source files using tree-sitter AST parsing. Extracts symbols and import edges for dependency tracking. Supports Python (.py), JS (.js/.jsx), TS (.ts/.tsx), Go (.go), Bash (.sh), Fish (.fish). Automatically deduplicates on re-index.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path":         map[string]string{"type": "string", "description": "Absolute path to file or directory to index"},
					"project_path": map[string]string{"type": "string", "description": "Project root path for grouping indexed symbols"},
				},
				"required": []string{"path", "project_path"},
			},
		},
		{
			Name:        "index_status",
			Description: "Get statistics about indexed symbols in a project. Returns total files and symbol count. Use to check if a project needs indexing.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Project root path (optional, returns all if omitted)"},
				},
			},
		},
		{
			Name:        "get_impact_graph",
			Description: "Find the blast radius of a symbol. Returns files that import or depend on the given symbol, enabling impact analysis before making changes. Requires project to be indexed first.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"symbol":       map[string]string{"type": "string", "description": "Symbol name (function, class, type, or module name) to analyze"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
				},
				"required": []string{"symbol", "project_path"},
			},
		},
		{
			Name:        "reset_project",
			Description: "Reset (delete) indexed data for a specific project. Use this to clear stale index data for one project.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project to reset"},
				},
				"required": []string{"project_path"},
			},
		},
		{
			Name:        "reset_all",
			Description: "Reset (delete) ALL indexed data for ALL projects. Use with caution - this will clear the entire index database.",
			InputSchema: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
	}
}

func handleTools(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	resp := JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      1,
		Result:  map[string]interface{}{"tools": getTools()},
	}
	json.NewEncoder(w).Encode(resp)
}

func handleIndexStatus(projectPath string) string {
	stats, err := getIndexStats(projectPath)
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, err)
	}
	data, _ := json.Marshal(stats)
	return string(data)
}

func readSourceRange(file string, startLine, endLine int, cache map[string][]string) string {
	if _, ok := cache[file]; !ok {
		content, err := os.ReadFile(file)
		if err != nil {
			return ""
		}
		cache[file] = strings.Split(string(content), "\n")
	}
	lines := cache[file]
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return ""
	}
	return strings.Join(lines[startLine-1:endLine], "\n")
}

func handleGetContext(args map[string]interface{}, projectPath string) string {
	query, _ := args["query"].(string)
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	if query == "" {
		return `{"error": "query required"}`
	}

	terms := strings.Fields(strings.ToLower(query))

	type scoredResult struct {
		data  map[string]interface{}
		score float64
	}
	var scored []scoredResult

	ftsQuery := buildFTSQuery(terms)
	if ftsQuery != "" {
		rows, err := db.Query(`
			SELECT s.name, s.kind, s.file, s.start_line, s.end_line, f.rank
			FROM symbols_fts f
			JOIN symbols s ON f.rowid = s.id
			WHERE s.project_path = ? AND symbols_fts MATCH ?
			ORDER BY f.rank
			LIMIT 100`, projectPath, ftsQuery)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var name, kind, file string
				var startLine, endLine int
				var rank float64
				rows.Scan(&name, &kind, &file, &startLine, &endLine, &rank)
				scored = append(scored, scoredResult{
					data: map[string]interface{}{
						"name": name, "kind": kind, "file": file,
						"start_line": startLine, "end_line": endLine,
					},
					score: -rank,
				})
			}
		}
	}

	if len(scored) == 0 {
		var conditions []string
		var sqlArgs []interface{}
		sqlArgs = append(sqlArgs, projectPath)
		for _, term := range terms {
			pattern := "%" + term + "%"
			conditions = append(conditions, "(LOWER(name) LIKE ? OR LOWER(fqn) LIKE ? OR LOWER(code) LIKE ?)")
			sqlArgs = append(sqlArgs, pattern, pattern, pattern)
		}
		where := "project_path = ?"
		if len(conditions) > 0 {
			where += " AND (" + strings.Join(conditions, " OR ") + ")"
		}
		rows, err := db.Query("SELECT name, kind, file, start_line, end_line FROM symbols WHERE "+where+" LIMIT 100", sqlArgs...)
		if err != nil {
			return fmt.Sprintf(`{"error": "%s"}`, err.Error())
		}
		defer rows.Close()
		for rows.Next() {
			var name, kind, file string
			var startLine, endLine int
			rows.Scan(&name, &kind, &file, &startLine, &endLine)
			s := 0.0
			nameLower := strings.ToLower(name)
			for _, t := range terms {
				if nameLower == t {
					s += 10
				} else if strings.HasPrefix(nameLower, t) {
					s += 5
				} else if strings.Contains(nameLower, t) {
					s += 3
				} else {
					s += 1
				}
			}
			scored = append(scored, scoredResult{
				data: map[string]interface{}{
					"name": name, "kind": kind, "file": file,
					"start_line": startLine, "end_line": endLine,
				},
				score: s,
			})
		}
		sort.Slice(scored, func(i, j int) bool {
			return scored[i].score > scored[j].score
		})
	}

	limit := 30
	if len(scored) < limit {
		limit = len(scored)
	}

	fileCache := map[string][]string{}
	results := make([]map[string]interface{}, limit)
	for i := 0; i < limit; i++ {
		results[i] = scored[i].data
		file, _ := results[i]["file"].(string)
		startLine, _ := results[i]["start_line"].(int)
		endLine, _ := results[i]["end_line"].(int)
		if src := readSourceRange(file, startLine, endLine, fileCache); src != "" {
			results[i]["source"] = src
		}
	}

	data, _ := json.Marshal(map[string]interface{}{"query": query, "results": results})
	return string(data)
}

func buildFTSQuery(terms []string) string {
	if len(terms) == 0 {
		return ""
	}
	var parts []string
	for _, t := range terms {
		cleaned := strings.Map(func(r rune) rune {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
				return r
			}
			return -1
		}, t)
		if cleaned != "" {
			parts = append(parts, cleaned+"*")
		}
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " OR ")
}

func handleAPIStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")

	var s Stats
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	if pid != "" {
		db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), COALESCE(SUM(tokens_saved),0) FROM queries WHERE project_path = ?", pid).
			Scan(&s.TotalQueries, &s.TotalSessions, &s.TotalChars, &s.AvgDurationMs, &s.TotalTokensSaved)
		db.QueryRow("SELECT COUNT(*), COALESCE(SUM(tokens_saved),0) FROM queries WHERE timestamp >= ? AND timestamp < ? AND project_path = ?", todayStart, tomorrowStart, pid).
			Scan(&s.TodayQueries, &s.TodayTokensSaved)
	} else {
		db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), COALESCE(SUM(tokens_saved),0) FROM queries").
			Scan(&s.TotalQueries, &s.TotalSessions, &s.TotalChars, &s.AvgDurationMs, &s.TotalTokensSaved)
		db.QueryRow("SELECT COUNT(*), COALESCE(SUM(tokens_saved),0) FROM queries WHERE timestamp >= ? AND timestamp < ?", todayStart, tomorrowStart).
			Scan(&s.TodayQueries, &s.TodayTokensSaved)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s)
}

func handleAPITools(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.Query("SELECT tool_name, COUNT(*) FROM queries WHERE project_path = ? GROUP BY tool_name", pid)
	} else {
		rows, err = db.Query("SELECT tool_name, COUNT(*) FROM queries GROUP BY tool_name")
	}
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	tools := []map[string]interface{}{}
	for rows.Next() {
		var n string
		var c int
		rows.Scan(&n, &c)
		tools = append(tools, map[string]interface{}{"name": n, "count": c})
	}
	json.NewEncoder(w).Encode(tools)
}

func handleAPIRecent(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	lim := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := fmt.Sscanf(l, "%d", &lim); err != nil || parsed != 1 || lim <= 0 {
			lim = 50
		}
	}
	if lim > 500 {
		lim = 500
	}
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.Query("SELECT timestamp, tool_name, result_chars, duration_ms, project_path, COALESCE(error,'') FROM queries WHERE project_path = ? ORDER BY timestamp DESC LIMIT ?", pid, lim)
	} else {
		rows, err = db.Query("SELECT timestamp, tool_name, result_chars, duration_ms, project_path, COALESCE(error,'') FROM queries ORDER BY timestamp DESC LIMIT ?", lim)
	}
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	qs := []map[string]interface{}{}
	for rows.Next() {
		var t, n, pp, errMsg string
		var rc int
		var dm float64
		rows.Scan(&t, &n, &rc, &dm, &pp, &errMsg)
		qs = append(qs, map[string]interface{}{"timestamp": t, "tool_name": n, "result_chars": rc, "duration_ms": dm, "project_path": pp, "error": errMsg})
	}
	json.NewEncoder(w).Encode(qs)
}

func handleAPIProjects(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	type symCount struct {
		symbols int
		files   int
	}
	symCounts := map[string]symCount{}
	symRows, err := db.Query("SELECT project_path, COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path IS NOT NULL GROUP BY project_path")
	if err == nil {
		defer symRows.Close()
		for symRows.Next() {
			var pp string
			var sc symCount
			symRows.Scan(&pp, &sc.symbols, &sc.files)
			symCounts[pp] = sc
		}
	}
	rows, err := db.Query("SELECT DISTINCT project_path, COUNT(*) FROM queries WHERE project_path IS NOT NULL GROUP BY project_path")
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	var ps []map[string]interface{}
	for rows.Next() {
		var p string
		var c int
		rows.Scan(&p, &c)
		sc := symCounts[p]
		ps = append(ps, map[string]interface{}{
			"path": p, "name": filepath.Base(p), "query_count": c,
			"symbol_count": sc.symbols, "file_count": sc.files,
		})
	}
	if ps == nil {
		ps = []map[string]interface{}{}
	}
	json.NewEncoder(w).Encode(ps)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "healthy", "service": "ast-context-cache", "version": "1.0.0"})
}

func handleAPIReset(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}

	var req map[string]string
	json.NewDecoder(r.Body).Decode(&req)
	projectPath := req["project_path"]

	db.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	db.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")

	if projectPath == "all" {
		_, err := db.Exec("DELETE FROM symbols")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		db.Exec("DELETE FROM edges")
		db.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
		ensureFTSTriggers()
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "message": "All indexed data cleared"})
	} else if projectPath != "" {
		_, err := db.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		db.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
		db.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
		ensureFTSTriggers()
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "project_path": projectPath})
	} else {
		json.NewEncoder(w).Encode(map[string]string{"error": "project_path required"})
	}
}

func handleAPIDeleteProject(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}

	var req map[string]string
	json.NewDecoder(r.Body).Decode(&req)
	projectPath := req["project_path"]

	if projectPath == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "project_path required"})
		return
	}

	_, err := db.Exec("DELETE FROM queries WHERE project_path = ?", projectPath)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	db.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	db.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")
	db.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
	db.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
	db.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	ensureFTSTriggers()

	json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "project_path": projectPath})
}

type TimeSeriesPoint struct {
	Timestamp     string  `json:"timestamp"`
	Queries       int     `json:"queries"`
	TokensSaved   int     `json:"tokens_saved"`
	AvgDurationMs float64 `json:"avg_duration_ms"`
}

func handleAPITimeseries(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	interval := r.URL.Query().Get("interval")
	if interval == "" {
		interval = "daily"
	}
	days := 30
	if d := r.URL.Query().Get("days"); d != "" {
		fmt.Sscanf(d, "%d", &days)
	}
	if days < 1 {
		days = 1
	} else if days > 365 {
		days = 365
	}

	var format string
	if interval == "hourly" {
		format = "%Y-%m-%dT%H:00"
	} else {
		format = "%Y-%m-%d"
	}

	var rows *sql.Rows
	var err error

	if pid != "" {
		rows, err = db.Query(`
			SELECT strftime(?, timestamp) as period,
			       COUNT(*) as queries,
			       COALESCE(SUM(tokens_saved), 0) as tokens_saved,
			       COALESCE(AVG(duration_ms), 0) as avg_duration_ms
			FROM queries
			WHERE project_path = ? AND timestamp >= datetime('now', '-' || ? || ' days')
			GROUP BY period
			ORDER BY period ASC
		`, format, pid, days)
	} else {
		rows, err = db.Query(`
			SELECT strftime(?, timestamp) as period,
			       COUNT(*) as queries,
			       COALESCE(SUM(tokens_saved), 0) as tokens_saved,
			       COALESCE(AVG(duration_ms), 0) as avg_duration_ms
			FROM queries
			WHERE timestamp >= datetime('now', '-' || ? || ' days')
			GROUP BY period
			ORDER BY period ASC
		`, format, days)
	}

	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	defer rows.Close()

	var points []TimeSeriesPoint
	for rows.Next() {
		var p TimeSeriesPoint
		rows.Scan(&p.Timestamp, &p.Queries, &p.TokensSaved, &p.AvgDurationMs)
		points = append(points, p)
	}

	if points == nil {
		points = []TimeSeriesPoint{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(points)
}

func handleAPIIndexStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")

	var totalSymbols, totalFiles, totalEdges int
	if pid != "" {
		db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", pid).Scan(&totalSymbols, &totalFiles)
		db.QueryRow("SELECT COUNT(*) FROM edges WHERE project_path = ?", pid).Scan(&totalEdges)
	} else {
		db.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&totalSymbols, &totalFiles)
		db.QueryRow("SELECT COUNT(*) FROM edges").Scan(&totalEdges)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_symbols": totalSymbols,
		"total_files":   totalFiles,
		"total_edges":   totalEdges,
	})
}

func handleAPISymbolKinds(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")

	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.Query("SELECT kind, COUNT(*) as count FROM symbols WHERE project_path = ? GROUP BY kind ORDER BY count DESC", pid)
	} else {
		rows, err = db.Query("SELECT kind, COUNT(*) as count FROM symbols GROUP BY kind ORDER BY count DESC")
	}
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()

	kinds := []map[string]interface{}{}
	for rows.Next() {
		var kind string
		var count int
		rows.Scan(&kind, &count)
		kinds = append(kinds, map[string]interface{}{"kind": kind, "count": count})
	}
	json.NewEncoder(w).Encode(kinds)
}

func handleAPILanguageStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")

	q := `SELECT
		CASE
			WHEN file LIKE '%.py' THEN 'Python'
			WHEN file LIKE '%.go' THEN 'Go'
			WHEN file LIKE '%.js' THEN 'JavaScript'
			WHEN file LIKE '%.jsx' THEN 'JSX'
			WHEN file LIKE '%.ts' THEN 'TypeScript'
			WHEN file LIKE '%.tsx' THEN 'TSX'
			WHEN file LIKE '%.sh' THEN 'Bash'
			WHEN file LIKE '%.fish' THEN 'Fish'
			ELSE 'Other'
		END as language,
		COUNT(DISTINCT file) as files,
		COUNT(*) as symbols
	FROM symbols`
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.Query(q+" WHERE project_path = ? GROUP BY language ORDER BY symbols DESC", pid)
	} else {
		rows, err = db.Query(q + " GROUP BY language ORDER BY symbols DESC")
	}
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()

	langs := []map[string]interface{}{}
	for rows.Next() {
		var lang string
		var files, symbols int
		rows.Scan(&lang, &files, &symbols)
		langs = append(langs, map[string]interface{}{"language": lang, "files": files, "symbols": symbols})
	}
	json.NewEncoder(w).Encode(langs)
}

func handleAPITopImports(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")

	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.Query("SELECT target, COUNT(*) as count FROM edges WHERE project_path = ? GROUP BY target ORDER BY count DESC LIMIT 20", pid)
	} else {
		rows, err = db.Query("SELECT target, COUNT(*) as count FROM edges GROUP BY target ORDER BY count DESC LIMIT 20")
	}
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()

	imports := []map[string]interface{}{}
	for rows.Next() {
		var target string
		var count int
		rows.Scan(&target, &count)
		imports = append(imports, map[string]interface{}{"target": target, "count": count})
	}
	json.NewEncoder(w).Encode(imports)
}

func handleAPIWatcherStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	watcherMu.Lock()
	watchers := []map[string]interface{}{}
	for project := range activeWatchers {
		watchers = append(watchers, map[string]interface{}{
			"project_path": project,
			"active":       true,
		})
	}
	watcherMu.Unlock()

	json.NewEncoder(w).Encode(map[string]interface{}{
		"watchers":     watchers,
		"total_active": len(watchers),
	})
}

func serveStatic(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	if path == "/" {
		path = "/index.html"
	}
	fp := filepath.Join(frontendDir, strings.TrimPrefix(path, "/"))
	if _, err := os.Stat(fp); os.IsNotExist(err) {
		fp = filepath.Join(frontendDir, "index.html")
	}
	ct := "text/plain"
	if strings.HasSuffix(fp, ".html") {
		ct = "text/html"
	} else if strings.HasSuffix(fp, ".css") {
		ct = "text/css"
	} else if strings.HasSuffix(fp, ".js") {
		ct = "application/javascript"
	}
	data, _ := os.ReadFile(fp)
	w.Header().Set("Content-Type", ct)
	w.Write(data)
}

func shouldSkipDir(name string) bool {
	return strings.HasPrefix(name, ".") || skipDirs[name]
}

var (
	watcherMu      sync.Mutex
	activeWatchers = map[string]*fsnotify.Watcher{}
	debounceMu     sync.Mutex
	debounceTimers = map[string]*time.Timer{}
	skipDirs = map[string]bool{
		"node_modules": true, "vendor": true, "venv": true, "env": true,
		"__pycache__": true, "dist": true, "build": true, ".astcache": true,
		"target": true, "bower_components": true, "third_party": true,
		"site-packages": true, "coverage": true, "tmp": true,
		"pkg": true, "Pods": true,
	}
)

func startWatcher(projectPath string) {
	watcherMu.Lock()
	if _, exists := activeWatchers[projectPath]; exists {
		watcherMu.Unlock()
		return
	}
	w, err := fsnotify.NewWatcher()
	if err != nil {
		watcherMu.Unlock()
		log.Printf("Watcher error for %s: %v", projectPath, err)
		return
	}
	activeWatchers[projectPath] = w
	watcherMu.Unlock()

	filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if shouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			w.Add(path)
		}
		return nil
	})

	go func() {
		for {
			select {
			case event, ok := <-w.Events:
				if !ok {
					return
				}
				handleFSEvent(event, projectPath, w)
			case err, ok := <-w.Errors:
				if !ok {
					return
				}
				log.Printf("Watcher error: %v", err)
			}
		}
	}()

	log.Printf("File watcher started for %s", projectPath)
}

func handleFSEvent(event fsnotify.Event, projectPath string, w *fsnotify.Watcher) {
	path := event.Name

	if info, err := os.Stat(path); err == nil && info.IsDir() {
		if event.Has(fsnotify.Create) && !shouldSkipDir(info.Name()) {
			w.Add(path)
		}
		return
	}

	if !isCodeFile(path) {
		return
	}

	removed := event.Has(fsnotify.Remove) || event.Has(fsnotify.Rename)

	debounceMu.Lock()
	if t, ok := debounceTimers[path]; ok {
		t.Stop()
	}
	debounceTimers[path] = time.AfterFunc(500*time.Millisecond, func() {
		start := time.Now()
		if removed {
			db.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", path, projectPath)
			db.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", path, projectPath)
			log.Printf("Removed symbols for deleted file: %s", path)
			logQuery("file_watcher", map[string]interface{}{"event": "delete", "file": path}, 0, 0, 0, 0,
				float64(time.Since(start).Milliseconds()), projectPath, "")
		} else {
			n, err := indexFile(path, projectPath)
			if err == nil {
				log.Printf("Re-indexed %s: %d symbols", path, n)
				resultJSON, _ := json.Marshal(map[string]interface{}{"file": path, "symbols": n})
				logQuery("file_watcher", map[string]interface{}{"event": "reindex", "file": path}, len(resultJSON), 0, 0, 0,
					float64(time.Since(start).Milliseconds()), projectPath, "")
			}
		}
		debounceMu.Lock()
		delete(debounceTimers, path)
		debounceMu.Unlock()
	})
	debounceMu.Unlock()
}

func main() {
	log.Println("Initializing...")

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)
	frontendDir = filepath.Join(exeDir, "dist")

	if err := initDB(); err != nil {
		log.Fatalf("DB error: %v", err)
	}

	go func() {
		for range time.NewTicker(5 * time.Minute).C {
			db.Exec(`PRAGMA wal_checkpoint(TRUNCATE)`)
		}
	}()

	restoreRows, err := db.Query("SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != ''")
	if err == nil {
		for restoreRows.Next() {
			var pp string
			restoreRows.Scan(&pp)
			if info, sErr := os.Stat(pp); sErr == nil && info.IsDir() {
				go startWatcher(pp)
			}
		}
		restoreRows.Close()
	}

	mcpMux := http.NewServeMux()
	mcpMux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method == "GET" {
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      1,
				Result:  map[string]interface{}{"tools": getTools()},
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
			// MCP handshake - return server capabilities
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
			// Notification - just acknowledge, no response needed
			// MCP clients send this after initialize
			return
		case "tools/list":
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Result:  map[string]interface{}{"tools": getTools()},
			})
		case "tools/call":
			start := time.Now()
			args := rpcReq.Params
			if args == nil {
				args = make(map[string]any)
			}

			toolName := ""
			if name, ok := args["name"].(string); ok {
				toolName = name
			}

			// Get arguments - could be nested in "arguments" or at top level
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
						var err error
						if info.IsDir() {
							n, err = indexDirectory(path, projectPath)
							if err == nil {
								go startWatcher(projectPath)
							}
						} else {
							n, err = indexFile(path, projectPath)
						}
						if err != nil {
							result = map[string]string{"error": err.Error()}
						} else {
							result = map[string]int{"indexed": n}
						}
					}
				}
			case "index_status":
				result, _ = getIndexStats(projectPath)
			case "get_context_capsule":
				query := ""
				if q, ok := toolArgs["query"].(string); ok {
					query = q
				}
				contextStr := handleGetContext(map[string]interface{}{"query": query}, projectPath)

				inputTokens := estimateTokens(query)
				outputTokens := estimateTokens(contextStr)
				const baselineContextTokens = 4000
				tokensSaved := baselineContextTokens - outputTokens
				if tokensSaved < 0 {
					tokensSaved = 0
				}

				logQuery(toolName, args, len(contextStr), inputTokens, outputTokens, tokensSaved, float64(time.Since(start).Milliseconds()), projectPath, "")

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
				resultStr := handleImpactGraph(map[string]interface{}{"symbol": sym}, projectPath)
				result = json.RawMessage(resultStr)
			case "reset_project":
				pp := ""
				if p, ok := toolArgs["project_path"].(string); ok {
					pp = p
				}
				if pp == "" {
					result = map[string]string{"error": "project_path required"}
				} else {
					_, err := db.Exec("DELETE FROM symbols WHERE project_path = ?", pp)
					if err != nil {
						result = map[string]string{"error": err.Error()}
					} else {
						db.Exec("DELETE FROM edges WHERE project_path = ?", pp)
						result = map[string]string{"status": "deleted", "project_path": pp}
					}
				}
			case "reset_all":
				_, err := db.Exec("DELETE FROM symbols")
				if err != nil {
					result = map[string]string{"error": err.Error()}
				} else {
					db.Exec("DELETE FROM edges")
					result = map[string]string{"status": "deleted", "message": "All indexed data cleared"}
				}
			default:
				result = map[string]string{"error": "not implemented: " + toolName}
			}

			if toolName != "get_context_capsule" {
				resultJSON, _ := json.Marshal(result)
				logQuery(toolName, args, len(resultJSON), 0, 0, 0, float64(time.Since(start).Milliseconds()), projectPath, "")
			}
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Result:  result,
			})
		default:
			json.NewEncoder(w).Encode(JSONRPCResponse{
				JSONRPC: JSONRPCVersion,
				ID:      rpcReq.ID,
				Error:   &JSONRPCError{Code: MethodNotFound, Message: "Unknown method: " + rpcReq.Method},
			})
		}
	})
	mcpMux.HandleFunc("/health", handleHealth)
	mcpMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") || strings.HasPrefix(r.URL.Path, "/mcp") {
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"service": "AST MCP", "dashboard": fmt.Sprintf("http://localhost:%d", DASHBOARD_PORT)})
	})

	dashMux := http.NewServeMux()
	dashMux.HandleFunc("/api/stats", handleAPIStats)
	dashMux.HandleFunc("/api/tools", handleAPITools)
	dashMux.HandleFunc("/api/recent", handleAPIRecent)
	dashMux.HandleFunc("/api/projects", handleAPIProjects)
	dashMux.HandleFunc("/api/reset", handleAPIReset)
	dashMux.HandleFunc("/api/delete", handleAPIDeleteProject)
	dashMux.HandleFunc("/api/timeseries", handleAPITimeseries)
	dashMux.HandleFunc("/api/index-stats", handleAPIIndexStats)
	dashMux.HandleFunc("/api/symbol-kinds", handleAPISymbolKinds)
	dashMux.HandleFunc("/api/language-stats", handleAPILanguageStats)
	dashMux.HandleFunc("/api/top-imports", handleAPITopImports)
	dashMux.HandleFunc("/api/watcher-status", handleAPIWatcherStatus)
	dashMux.HandleFunc("/", serveStatic)

	go func() {
		addr := fmt.Sprintf(":%d", MCP_PORT)
		log.Printf("MCP: http://localhost%s/mcp", addr)
		log.Fatal(http.ListenAndServe(addr, mcpMux))
	}()

	addr := fmt.Sprintf(":%d", DASHBOARD_PORT)
	log.Printf("Dashboard: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, dashMux))
}
