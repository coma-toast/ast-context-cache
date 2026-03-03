package mcp

func GetTools() []Tool {
	return []Tool{
		{
			Name:        "get_context_capsule",
			Description: "Search indexed code symbols using hybrid BM25+vector search. Returns matching functions, classes, types with file paths, line ranges. Supports mode: 'full' (source code), 'skeleton' (signatures only, ~90% token reduction), 'summary' (cached summaries, ~94% token reduction), 'auto' (full for top 3 results, skeleton for rest). Supports Python, JS/TS, Go, Bash, Fish.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query":        map[string]string{"type": "string", "description": "Search query (function name, class name, type name, or keywords)"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"mode":         map[string]string{"type": "string", "description": "Response mode: 'full' (default, returns source), 'skeleton' (signatures only), 'summary' (cached summaries), 'auto' (full for top 3 results, skeleton for rest — best balance of detail and token efficiency)"},
					"session_id":   map[string]string{"type": "string", "description": "Session ID for dedup. If provided, symbols already returned in this session are skipped."},
				"token_budget": map[string]string{"type": "integer", "description": "Max tokens to return (default 4000). Results are packed greedily by score until budget is exhausted."},
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
			Name:        "cache_summary",
			Description: "Store a summary for a file or symbol. LLMs call this to 'write back' what they learned about code. Summaries are cached and used by get_context in summary mode to dramatically reduce tokens.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"file":         map[string]string{"type": "string", "description": "Absolute path to the file"},
					"symbol":       map[string]string{"type": "string", "description": "Symbol name (optional, omit for file-level summary)"},
					"summary":      map[string]string{"type": "string", "description": "The summary text to cache"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
				},
				"required": []string{"file", "summary", "project_path"},
			},
		},
		{
			Name:        "search_semantic",
			Description: "Semantic vector search over indexed code symbols. Finds symbols by meaning, not just text matching. Requires project to be indexed with embeddings. Returns ranked results with similarity scores and source code.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query":        map[string]string{"type": "string", "description": "Natural language search query (e.g. 'function that handles user authentication')"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"limit":        map[string]string{"type": "integer", "description": "Max results to return (default 10)"},
					"doc_type":     map[string]string{"type": "string", "description": "Filter by document type: 'code', 'doc', etc. (optional)"},
					"session_id":   map[string]string{"type": "string", "description": "Session ID for dedup. If provided, symbols already returned in this session are skipped."},
					"token_budget": map[string]string{"type": "integer", "description": "Max tokens to return. Results packed greedily by score until budget exhausted."},
				},
				"required": []string{"query", "project_path"},
			},
		},
		{
			Name:        "get_project_map",
			Description: "Get a hierarchical overview of an indexed project's structure. Returns directories, files, and symbols at configurable depth. Use to understand a project in ~200 tokens instead of reading every file.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"depth":        map[string]string{"type": "integer", "description": "Detail level: 1=dirs only, 2=dirs+files (default), 3=files+symbols"},
				},
				"required": []string{"project_path"},
			},
		},
		{
			Name:        "get_file_context",
			Description: "Get all symbols in a specific file with mode selection. Smarter than raw file read -- returns structured, mode-aware context. Modes: 'full' (source code), 'skeleton' (signatures), 'summary' (cached summaries).",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"file":         map[string]string{"type": "string", "description": "Absolute path to the file"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"mode":         map[string]string{"type": "string", "description": "Response mode: 'full' (default), 'skeleton', 'summary'"},
					"session_id":   map[string]string{"type": "string", "description": "Session ID for dedup."},
					"token_budget": map[string]string{"type": "integer", "description": "Max tokens to return."},
				},
				"required": []string{"file", "project_path"},
			},
		},
		{
			Name:        "sync_remote",
			Description: "Push or pull vectors to/from a remote Milvus vector database. Use for backup and cross-machine sync. Configured via REMOTE_VECTORDB_URL env var (default: https://ai.jasondale.me/vectordb/mcp).",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Project path to sync (optional, syncs all if omitted)"},
					"direction":    map[string]string{"type": "string", "description": "'push' (local->remote, default) or 'pull' (remote->local)"},
					"collection":   map[string]string{"type": "string", "description": "Remote collection name (default: configsync_docs)"},
				},
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
