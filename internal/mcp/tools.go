package mcp

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type ToolConfig struct {
	Enabled     bool   `json:"enabled"`
	Tier        Tier   `json:"tier"`
	Description string `json:"description"`
}

func getToolsConfigPath() string {
	if p := os.Getenv("AST_MCP_TOOLS_CONFIG"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".astcache", "tools.json")
}

func LoadToolConfigs() map[string]*ToolConfig {
	path := getToolsConfigPath()
	data, err := os.ReadFile(path)
	if err != nil {
		return make(map[string]*ToolConfig)
	}
	var configs map[string]*ToolConfig
	if err := json.Unmarshal(data, &configs); err != nil {
		return make(map[string]*ToolConfig)
	}
	for _, c := range configs {
		if c != nil && c.Tier != "" {
			c.Tier = ParseTier(string(c.Tier))
		}
	}
	return configs
}

func SaveToolConfigs(configs map[string]*ToolConfig) error {
	path := getToolsConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(configs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

type toolDenyReason int

const (
	denyNone toolDenyReason = iota
	denyUnknown
	denyDisabled
	denyTier
	denyCodeMode
)

func effectiveRequiredTier(t Tool, cfg ServerConfig) Tier {
	if conf, ok := cfg.ToolConfigs[t.Name]; ok && conf.Tier != "" {
		return conf.Tier
	}
	return t.Tier
}

func toolAccess(t Tool, cfg ServerConfig) (bool, toolDenyReason) {
	if conf, ok := cfg.ToolConfigs[t.Name]; ok && !conf.Enabled {
		return false, denyDisabled
	}
	if !TierIncludes(cfg.ActiveTier, effectiveRequiredTier(t, cfg)) {
		return false, denyTier
	}
	if t.Name == "execute_code" && !cfg.CodeMode {
		return false, denyCodeMode
	}
	return true, denyNone
}

// ToolDenyMessage explains why a tool call was rejected.
func ToolDenyMessage(toolName string, cfg ServerConfig, reason toolDenyReason) string {
	switch reason {
	case denyDisabled:
		return "tool disabled in tools config: " + toolName
	case denyTier:
		return "tool not available at server tier " + string(cfg.ActiveTier) + ": " + toolName
	case denyCodeMode:
		return "execute_code disabled (set AST_MCP_CODE_MODE or enable in policy): " + toolName
	case denyUnknown:
		return "unknown tool: " + toolName
	default:
		return "tool not allowed: " + toolName
	}
}

func toolAccessByName(toolName string, cfg ServerConfig) (bool, toolDenyReason) {
	for _, t := range GetTools() {
		if t.Name == toolName {
			return toolAccess(t, cfg)
		}
	}
	return false, denyUnknown
}

func GetTools() []Tool {
	return []Tool{
		{
			Name:        "get_context_capsule",
			Description: "Search indexed code symbols using hybrid BM25+vector search. Returns matching functions, classes, types with file paths, line ranges. Supports mode: 'full' (source code), 'skeleton' (signatures only, ~90% token reduction), 'summary' (cached summaries, ~94% token reduction), 'auto' (full for top 3 results, skeleton for rest). Supports Python, JS/TS, Go, Bash, Fish.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query":         map[string]string{"type": "string", "description": "Search query (function name, class name, type name, or keywords)"},
					"project_path":  map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"mode":          map[string]string{"type": "string", "description": "Response mode: 'auto' (default — full for top hits, skeleton for rest), 'skeleton', 'summary' (cached summaries), 'full'"},
					"session_id":    map[string]string{"type": "string", "description": "Session ID for dedup. If provided, symbols already returned in this session are skipped."},
					"token_budget":  map[string]string{"type": "integer", "description": "Max tokens to return (default 4000). Results are packed greedily by score until budget is exhausted."},
					"path_prefix":   map[string]string{"type": "string", "description": "Optional: only symbols under this path (project-relative, e.g. internal/mcp) or absolute path prefix."},
					"language":      map[string]string{"type": "string", "description": "Optional: filter by language (go, python, typescript, javascript, rust, ...). Uses file extensions."},
					"kinds":         map[string]string{"type": "string", "description": "Optional: comma-separated symbol kinds to include (e.g. function,method)."},
					"kind":          map[string]string{"type": "string", "description": "Optional: single symbol kind filter (same as one entry in kinds)."},
				},
				"required": []string{"query", "project_path"},
			},
			Tier:     TierCore,
			ReadOnly: true,
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
			Tier: TierExtended,
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
			Tier:     TierCore,
			ReadOnly: true,
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
			Tier:     TierCore,
			ReadOnly: true,
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
			Tier: TierExtended,
		},
		{
			Name:        "search_semantic",
			Description: "Semantic vector search over indexed code symbols. Finds symbols by meaning, not just text matching. Supports mode (default skeleton), session_id, and token_budget like get_context_capsule.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query":         map[string]string{"type": "string", "description": "Natural language search query (e.g. 'function that handles user authentication')"},
					"project_path":  map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"limit":         map[string]string{"type": "integer", "description": "Max results to return (default 10)"},
					"doc_type":      map[string]string{"type": "string", "description": "Filter by document type: 'code', 'doc', etc. (optional)"},
					"session_id":    map[string]string{"type": "string", "description": "Session ID for dedup. If provided, symbols already returned in this session are skipped."},
					"token_budget":  map[string]string{"type": "integer", "description": "Max tokens to return. Results packed greedily by score until budget exhausted."},
					"path_prefix":   map[string]string{"type": "string", "description": "Optional: only symbols under this path (project-relative or absolute prefix)."},
					"language":      map[string]string{"type": "string", "description": "Optional: filter by language (go, python, typescript, ...)."},
					"kinds":         map[string]string{"type": "string", "description": "Optional: comma-separated symbol kinds to include."},
					"kind":          map[string]string{"type": "string", "description": "Optional: single symbol kind filter."},
				},
				"required": []string{"query", "project_path"},
			},
			Tier:     TierCore,
			ReadOnly: true,
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
			Tier:     TierCore,
			ReadOnly: true,
		},
		{
			Name:        "get_file_context",
			Description: "Get all symbols in a specific file with mode-aware output. Returns function signatures, source code, or cached summaries for every symbol in the file. Supports mode: 'full' (source code), 'skeleton' (signatures only, ~90% token reduction), 'summary' (cached summaries, ~94% token reduction).",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"file":         map[string]string{"type": "string", "description": "Absolute path to the file"},
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"mode":         map[string]string{"type": "string", "description": "Response mode: 'skeleton' (default), 'full', 'summary' (cached summaries), 'auto'"},
					"session_id":   map[string]string{"type": "string", "description": "Session ID for dedup — skip symbols already returned in this session"},
					"token_budget": map[string]string{"type": "integer", "description": "Max tokens for symbol list (optional)"},
				},
				"required": []string{"file", "project_path"},
			},
			Tier:     TierCore,
			ReadOnly: true,
		},
		{
			Name:        "analyze_dead_code",
			Description: "Find unused functions, classes, and imports in indexed code. Identifies symbols that are never called or imported, helping to identify code that can be removed.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"kind":         map[string]string{"type": "string", "description": "Type of symbols to check: 'function', 'class', 'method', 'import' (default: all)"},
				},
				"required": []string{"project_path"},
			},
			Tier:     TierExtended,
			ReadOnly: true,
		},
		{
			Name:        "analyze_complexity",
			Description: "Calculate cyclomatic complexity for functions and methods. Returns complexity scores to identify potentially hard-to-maintain code that may need refactoring.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"threshold":    map[string]string{"type": "integer", "description": "Minimum complexity to report (default: 10)"},
					"limit":        map[string]string{"type": "integer", "description": "Max results to return (default: 20)"},
				},
				"required": []string{"project_path"},
			},
			Tier:     TierExtended,
			ReadOnly: true,
		},
		{
			Name:        "execute_code",
			Description: "Execute JavaScript code in a sandboxed environment against search results. The LLM writes processing code that runs locally - only the output enters context, saving 65-99% tokens. Data is injected as 'DATA' variable.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"code":    map[string]string{"type": "string", "description": "JavaScript code to execute. Results available as 'DATA' variable (array of search results). Return array for output."},
					"data":    map[string]string{"type": "string", "description": "JSON stringified data to process (typically from a previous search)"},
					"timeout": map[string]string{"type": "integer", "description": "Execution timeout in seconds (default: 5)"},
				},
				"required": []string{"code", "data"},
			},
			Tier: TierComplete,
		},
		{
			Name:        "export_bundle",
			Description: "Export indexed code as a portable bundle file. Bundles can be shared or imported on other machines without re-indexing.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]string{"type": "string", "description": "Absolute path to the project to export"},
					"output_path":  map[string]string{"type": "string", "description": "Output file path for the bundle (.astbundle)"},
				},
				"required": []string{"project_path", "output_path"},
			},
			Tier:     TierExtended,
			ReadOnly: true,
		},
		{
			Name:        "import_bundle",
			Description: "Import a previously exported code bundle. Loads all symbols, edges, and summaries without needing to re-index.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"bundle_path": map[string]string{"type": "string", "description": "Path to the .astbundle file to import"},
				},
				"required": []string{"bundle_path"},
			},
			Tier: TierExtended,
		},
		{
			Name:        "search_docs",
			Description: "Search locally cached documentation by title or content (FTS). Try this before WebFetch or web search for library/framework docs.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]string{"type": "string", "description": "Search query for documentation"},
					"limit": map[string]string{"type": "integer", "description": "Max results to return (default 10)"},
				},
				"required": []string{"query"},
			},
			Tier:     TierCore,
			ReadOnly: true,
		},
		{
			Name:        "fetch_doc",
			Description: "Fetch a documentation URL, register it in the local doc cache, and return stored entries. Prefer this over WebFetch for external library/framework docs so future search_docs queries hit the cache.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name":           map[string]string{"type": "string", "description": "Name of the documentation (e.g. 'React', 'Express')"},
					"type":           map[string]string{"type": "string", "description": "Documentation type: 'markdown', 'html', 'json'"},
					"url":            map[string]string{"type": "string", "description": "URL to fetch documentation from"},
					"version":        map[string]string{"type": "string", "description": "Version of the documentation (optional)"},
					"force_refresh":  map[string]string{"type": "boolean", "description": "Re-fetch from URL even if cached content is fresh (default false)"},
				},
				"required": []string{"name", "type", "url"},
			},
			Tier: TierExtended,
		},
		{
			Name:        "add_doc_source",
			Description: "Track a documentation URL for background caching (async fetch). Use fetch_doc when you need content in the same response; use this to register URLs without waiting.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name":    map[string]string{"type": "string", "description": "Name of the documentation (e.g. 'React', 'Express')"},
					"type":    map[string]string{"type": "string", "description": "Documentation type: 'markdown', 'html', 'json'"},
					"url":     map[string]string{"type": "string", "description": "URL to fetch documentation from"},
					"version": map[string]string{"type": "string", "description": "Version of the documentation (optional)"},
				},
				"required": []string{"name", "type", "url"},
			},
			Tier: TierExtended,
		},
		{
			Name:        "remove_doc_source",
			Description: "Remove a documentation source and all its cached content.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"id": map[string]string{"type": "integer", "description": "ID of the doc source to remove"},
				},
				"required": []string{"id"},
			},
			Tier: TierExtended,
		},
		{
			Name:        "list_doc_sources",
			Description: "List all tracked documentation sources with their last update time.",
			InputSchema: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
			Tier:     TierCore,
			ReadOnly: true,
		},
		{
			Name:        "update_doc_source",
			Description: "Manually re-fetch a tracked documentation source from its URL.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"id": map[string]string{"type": "integer", "description": "ID of the doc source to update"},
				},
				"required": []string{"id"},
			},
			Tier: TierExtended,
		},
		{
			Name:        "retrieve",
			Description: "RAG-style retrieval: hybrid search across code + docs, reranks results, assembles context within token budget. Returns formatted context ready for LLM consumption. Supports markdown, xml, json output formats.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query":          map[string]string{"type": "string", "description": "Natural language query to retrieve context for"},
					"project_path":   map[string]string{"type": "string", "description": "Absolute path to the project root"},
					"limit":          map[string]string{"type": "integer", "description": "Max results to return (default 10)"},
					"token_budget":   map[string]string{"type": "integer", "description": "Max tokens for assembled context (default 4000)"},
					"include_docs":   map[string]string{"type": "boolean", "description": "Include documentation sources (default true)"},
					"include_source": map[string]string{"type": "boolean", "description": "Include full source code in code chunks (default false — skeleton/signatures only)"},
					"mode":           map[string]string{"type": "string", "description": "Code chunk mode when include_source is false: 'skeleton' (default), 'auto', 'full'"},
					"session_id":     map[string]string{"type": "string", "description": "Session ID for dedup — skip symbols already returned"},
					"format":         map[string]string{"type": "string", "description": "Output format: 'markdown', 'xml', 'json' (default 'markdown')"},
					"path_prefix":    map[string]string{"type": "string", "description": "Optional: only code symbols under this path (project-relative or absolute prefix)."},
					"language":       map[string]string{"type": "string", "description": "Optional: filter by language (go, typescript, ...)."},
					"kinds":          map[string]string{"type": "string", "description": "Optional: comma-separated symbol kinds to include."},
					"kind":           map[string]string{"type": "string", "description": "Optional: single symbol kind filter."},
				},
				"required": []string{"query", "project_path"},
			},
			Tier:     TierCore,
			ReadOnly: true,
		},
	}
}

// FilterTools returns only tools accessible at the given tier/config.
func FilterTools(cfg ServerConfig) []Tool {
	all := GetTools()
	filtered := make([]Tool, 0, len(all))
	for _, t := range all {
		ok, _ := toolAccess(t, cfg)
		if !ok {
			continue
		}
		out := t
		if conf, ok := cfg.ToolConfigs[t.Name]; ok && conf.Description != "" {
			out.Description = conf.Description
		}
		filtered = append(filtered, out)
	}
	return filtered
}

// IsToolAllowed checks if a specific tool name is permitted under the given config.
func IsToolAllowed(toolName string, cfg ServerConfig) bool {
	ok, _ := toolAccessByName(toolName, cfg)
	return ok
}

// GetPrompts returns LLM-optimized system prompts for efficient tool usage
func GetPrompts() []Prompt {
	return []Prompt{
		{
			Name:        "efficient-context-usage",
			Description: "Guidelines for token-efficient code context retrieval",
			Prompt: `# Efficient Code Context Usage Guide

## Token Optimization Strategies

### 1. Use 'auto' Mode (Recommended Default)
- Mode 'auto' returns full source for top 3 results, skeleton for rest
- This provides ~80% token savings while maintaining detail for key matches
- Only use 'full' when you need complete implementation details

### 2. Use 'skeleton' Mode for Exploration
- Use mode='skeleton' when exploring unfamiliar codebases
- Returns function signatures only (~90% token reduction)
- Perfect for understanding structure before diving into details

### 3. Use 'summary' Mode for Broad Context
- Use mode='summary' after exploring to get high-level overviews
- Requires calling cache_summary first to create summaries
- Provides ~94% token reduction

### 4. Leverage Session Deduplication
- Pass the same session_id on get_context_capsule, search_semantic, retrieve, and get_file_context
- Skipped symbols add to dedup_tokens_saved in the response and dashboard

### Token savings tracking
- tokens_saved = max(0, full_source_baseline - tokens_returned) + dedup_skips
- Tracked: get_context_capsule, get_file_context, search_semantic, retrieve (see tokens_saved, tokens_used, symbol_baseline_tokens in JSON)
- Not tracked: fetch_doc, search_docs, index_*, get_project_map — dashboard Tokens saved can be 0 on doc-only days
- mode=full saves ~nothing; use auto or skeleton for real savings

### 5. Use Token Budget Wisely
- Set token_budget to control response size
- Default is 4000 tokens - adjust based on need
- Tool stops adding results when budget is exhausted

### 6. Use Semantic Search for Intent
- search_semantic finds symbols by meaning, not just text
- Natural language queries: "function that handles auth"
- Great for exploratory searches

### 7. Cache Your Own Summaries
- Call cache_summary after understanding a file
- Future queries using mode='summary' will use your cached summaries
- This creates personalized, token-efficient context

### 8. Use get_project_map First
- For new projects, start with get_project_map (depth=1 or 2)
- Understand structure before diving into files
- Only ~200 tokens for full project overview

### 9. Use get_impact_graph for Change Analysis
- Before modifying code, call get_impact_graph
- Shows all files that depend on a symbol
- Helps understand blast radius of changes

### Recommended Workflow
1. get_project_map to understand structure
2. get_context_capsule with mode='auto' for initial search
3. Use skeleton mode for broad exploration
4. Cache summaries of key files
5. Use impact graph before making changes`,
		},
		{
			Name:        "context-mode-decisions",
			Description: "When to use each context retrieval mode",
			Prompt: `# Context Mode Selection Guide

## Mode Quick Reference

| Mode | Tokens | Use When |
|------|--------|----------|
| 'full' | 100% | Need complete implementation details |
| 'auto' | ~20% | Default - need balance of detail and efficiency |
| 'skeleton' | ~10% | Exploring structure, understanding signatures |
| 'summary' | ~6% | Already cached, need high-level overview |

## Decision Tree

### Initial Exploration of Unfamiliar Code
→ Use mode='skeleton' or mode='auto'
→ Get broad understanding without token bloat

### Deep Dive into Specific Function
→ Use mode='full' for the specific function
→ Keep token budget small (1000-2000)

### Understanding File Structure
→ Use get_file_context with mode='skeleton'
→ See all functions/classes at once

### Repeating Context in Same Session
→ session_id enables automatic deduplication
→ Same files won't be re-sent

### Broad Search Across Many Files
→ mode='auto' with high limit (20-30)
→ Get top results in full, rest as skeleton

### After Writing Code (Self-Documentation)
→ Call cache_summary to document what you learned
→ Future sessions can benefit from your summaries`,
		},
	}
}
