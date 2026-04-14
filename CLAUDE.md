# Code Context Instructions

# MCP Server for This Project

This project includes an MCP server (`ast-context-cache`) that provides efficient code search tools.
When researching this codebase, **always prefer using the MCP tools** over direct grep/read:
- Faster and more token-efficient than grep/read/glob
- Supports semantic search, doc caching, and smart deduplication
- Run `make run` to start the MCP server, then use the tools below

# Available Tools

## Core
- **get_context_capsule** - BM25+vector hybrid search with modes: full, skeleton, summary, auto
- **search_semantic** - Natural language search: "function that handles auth"
- **get_file_context** - All symbols in a file with mode-aware output (use instead of reading files)
- **get_project_map** - Project structure overview (~200 tokens at depth=2)
- **get_impact_graph** - Blast radius of a symbol before making changes
- **index_status** - Check if a project is indexed
- **search_docs** - Search cached library/framework documentation
- **retrieve** - RAG-style retrieval: hybrid search + reranking + context assembly (code + docs)

## Extended
- **index_files** - Index a file or directory (starts file watcher)
- **cache_summary** - Cache summaries for future queries
- **analyze_dead_code** - Find unused functions, classes, imports
- **analyze_complexity** - Find hard-to-maintain code by cyclomatic complexity
- **export_bundle** / **import_bundle** - Portable code bundles without re-indexing
- **add_doc_source** / **remove_doc_source** / **list_doc_sources** / **update_doc_source** - Track and cache external documentation

## Complete
- **execute_code** - Run JS in a sandbox against search results; only output enters context

# Efficient Code Context Usage Guide

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
- Always pass session_id to get_context_capsule
- This prevents re-sending files you've already seen in this conversation
- The tool tracks what's been returned and auto-skips duplicates

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

### 10. Use search_docs for Documentation
- Search cached library/framework docs instead of web searches
- Add sources with add_doc_source (supports markdown, html, json URLs)
- Doc sources auto-refresh every hour

### Recommended Workflow
1. get_project_map to understand structure
2. get_context_capsule with mode='auto' for initial search
3. get_file_context for all symbols in a specific file
4. Use skeleton mode for broad exploration
5. Cache summaries of key files
6. Use impact graph before making changes
7. Use search_docs for library/framework questions

### Optional search filters (get_context_capsule, search_semantic, retrieve)

- **path_prefix** — limit to symbols under a project-relative path (e.g. `internal/mcp`) or absolute prefix.
- **language** — `go`, `python`, `typescript`, `javascript`, `rust`, etc. (file extension filter).
- **kinds** / **kind** — comma-separated or single symbol kind (`function`, `method`, …).

### Pipeline observability

- **get_context_capsule** — response includes **pipeline**: `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse`.
- **retrieve** — **stats** adds the same hybrid counts plus `after_dedup`, `chunks_in_budget`, `tokens_est_all_chunks`, and stage timings (`code_retrieve_ms`, `docs_retrieve_ms`, `dedup_budget_ms`, `search_time_ms`).
