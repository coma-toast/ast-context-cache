// Package coverage reports which ast-context-cache features agent sessions actually use,
// from the usage.db queries table. Features is the single source of the tool -> feature
// grouping; the dashboard card, weekly digest and Prometheus gauges all read it.
package coverage

import "strings"

// FeatureSearch is the code-search feature; sessions that never use it are "search-less".
const FeatureSearch = "search"

// Feature groups the MCP tools that make up one capability.
type Feature struct {
	Name  string
	Label string
	Tools []string
}

// Features lists every feature in display order.
var Features = []Feature{
	{Name: FeatureSearch, Label: "Code search", Tools: []string{"get_context_capsule", "search_semantic", "retrieve", "get_file_context", "get_project_map", "check_symbol_exists"}},
	{Name: "impact", Label: "Impact analysis", Tools: []string{"get_impact_graph", "diff_impact", "check_deletion_safety"}},
	{Name: "virtual_context", Label: "Virtual context", Tools: []string{"store_context", "fetch_context", "list_context", "search_context", "edit_context", "flush_context", "get_pinned_note", "report_kv_repair_event"}},
	{Name: "memory", Label: "Structured memory", Tools: []string{"store_memory", "recall_memory", "forget_memory"}},
	{Name: "docs", Label: "Docs cache", Tools: []string{"search_docs", "fetch_doc", "list_doc_sources", "add_doc_source", "remove_doc_source", "update_doc_source"}},
	{Name: "code_mode", Label: "Code mode", Tools: []string{"execute_code"}},
	{Name: "handoff", Label: "Handoff", Tools: []string{"handoff", "open_handoff", "scratchpad"}},
	{Name: "context_fn", Label: "Context functions", Tools: []string{"define_context_fn", "apply_context_fn", "list_context_fns"}},
	{Name: "analysis", Label: "Code analysis", Tools: []string{"analyze_dead_code", "analyze_complexity"}},
	{Name: "index", Label: "Indexing", Tools: []string{"index_files", "index_status", "cache_summary"}},
}

// internalTools are query-log rows that are not agent tool calls.
var internalTools = []string{"file_watcher"}

var toolFeature = func() map[string]string {
	m := map[string]string{}
	for _, f := range Features {
		for _, t := range f.Tools {
			m[t] = f.Name
		}
	}
	return m
}()

// FeatureOf returns the feature a tool belongs to, or "" for an unmapped tool.
func FeatureOf(tool string) string {
	return toolFeature[tool]
}

// IsInternal reports whether a query-log tool name is internal (not an agent call).
func IsInternal(tool string) bool {
	for _, t := range internalTools {
		if t == tool {
			return true
		}
	}
	return false
}

// ToolsFor returns the tools of the named feature, or nil when it does not exist.
func ToolsFor(feature string) []string {
	for _, f := range Features {
		if f.Name == feature {
			return f.Tools
		}
	}
	return nil
}

// excludeInternalClause is a SQL predicate dropping internal rows.
func excludeInternalClause() string {
	return "tool_name NOT IN ('" + strings.Join(internalTools, "','") + "')"
}
