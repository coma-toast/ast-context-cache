package mcp

import (
	"fmt"
	"strings"
)

func ExampleFilterTools_core() {
	cfg := ServerConfig{ActiveTier: TierCore, CodeMode: false}
	tools := FilterTools(cfg)

	var names []string
	for _, t := range tools {
		names = append(names, t.Name)
	}
	fmt.Println(strings.Join(names, ", "))
	// Output: get_context_capsule, index_status, get_impact_graph, search_semantic, get_project_map, get_file_context, search_docs, list_doc_sources, retrieve
}

func ExampleFilterTools_extended() {
	cfg := ServerConfig{ActiveTier: TierExtended, CodeMode: false}
	tools := FilterTools(cfg)

	var names []string
	for _, t := range tools {
		names = append(names, t.Name)
	}
	fmt.Println(strings.Join(names, ", "))
	// Output: get_context_capsule, index_files, index_status, get_impact_graph, cache_summary, search_semantic, get_project_map, get_file_context, analyze_dead_code, analyze_complexity, export_bundle, import_bundle, search_docs, add_doc_source, remove_doc_source, list_doc_sources, update_doc_source, retrieve
}

func ExampleFilterTools_complete() {
	cfg := ServerConfig{ActiveTier: TierComplete, CodeMode: true}
	tools := FilterTools(cfg)

	var names []string
	for _, t := range tools {
		names = append(names, t.Name)
	}
	fmt.Println(strings.Join(names, ", "))
	// Output: get_context_capsule, index_files, index_status, get_impact_graph, cache_summary, search_semantic, get_project_map, get_file_context, analyze_dead_code, analyze_complexity, execute_code, export_bundle, import_bundle, search_docs, add_doc_source, remove_doc_source, list_doc_sources, update_doc_source, retrieve
}

func ExampleFilterTools_completeNoCodeMode() {
	cfg := ServerConfig{ActiveTier: TierComplete, CodeMode: false}
	tools := FilterTools(cfg)

	for _, t := range tools {
		if t.Name == "execute_code" {
			fmt.Println("execute_code found (unexpected)")
			return
		}
	}
	fmt.Println("execute_code excluded when CodeMode=false")
	// Output: execute_code excluded when CodeMode=false
}

func ExampleIsToolAllowed() {
	core := ServerConfig{ActiveTier: TierCore, CodeMode: false}
	full := ServerConfig{ActiveTier: TierComplete, CodeMode: true}

	fmt.Println("core+get_context_capsule:", IsToolAllowed("get_context_capsule", core))
	fmt.Println("core+index_files:", IsToolAllowed("index_files", core))
	fmt.Println("core+execute_code:", IsToolAllowed("execute_code", core))
	fmt.Println("complete+execute_code:", IsToolAllowed("execute_code", full))
	// Output:
	// core+get_context_capsule: true
	// core+index_files: false
	// core+execute_code: false
	// complete+execute_code: true
}

func ExampleTierIncludes() {
	fmt.Println("complete includes core:", TierIncludes(TierComplete, TierCore))
	fmt.Println("core includes extended:", TierIncludes(TierCore, TierExtended))
	fmt.Println("extended includes extended:", TierIncludes(TierExtended, TierExtended))
	// Output:
	// complete includes core: true
	// core includes extended: false
	// extended includes extended: true
}

// Example_tierAnnotation shows how to add a new tool with tier annotation.
//
// When adding a tool to GetTools(), include Tier and ReadOnly fields:
//
//	{
//	    Name:        "my_custom_tool",
//	    Description: "Does something useful",
//	    InputSchema: map[string]interface{}{...},
//	    Tier:        TierExtended,
//	    ReadOnly:    true,
//	}
//
// Tier assignment guide:
//   - TierCore:     Search, status, read-only queries (always safe)
//   - TierExtended: Indexing, caching, analysis (writes to DB but no exec)
//   - TierComplete: Code execution, sandbox operations (requires CodeMode=true for execute_code)
func Example_tierAnnotation() {
	tool := Tool{
		Name:        "my_custom_tool",
		Description: "Example tool demonstrating tier annotation",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"input": map[string]string{"type": "string", "description": "some input"},
			},
			"required": []string{"input"},
		},
		Tier:     TierExtended,
		ReadOnly: true,
	}

	cfg := ServerConfig{ActiveTier: TierCore}
	fmt.Println("allowed at core:", IsToolAllowed(tool.Name, cfg))

	cfg.ActiveTier = TierExtended
	fmt.Println("allowed at extended:", IsToolAllowed(tool.Name, cfg))
	// Output:
	// allowed at core: false
	// allowed at extended: false
}
