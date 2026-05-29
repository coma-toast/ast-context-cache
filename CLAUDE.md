# Code Context Instructions

# MCP Server for This Project

This project includes an MCP server (`ast-context-cache`) that provides efficient code search tools.
When researching this codebase, **always prefer using the MCP tools** over direct grep/read:
- Faster and more token-efficient than grep/read/glob
- Supports semantic search, doc caching, and smart deduplication
- Run `make run` to start the MCP server, then use the tools below (or `ast-mcp start` if you installed the shell function with `make install`).

**Optional launcher:** For a unified local MCP supervisor (start `ast-mcp`, merge MCP config, etc.), see the standalone [mcp-local](https://github.com/coma-toast/mcp-local) repository and its README. This repo does not include that binary.

**Tool tiers:** The host sets `AST_MCP_TIER` (`core` / `extended` / `complete`) and optional `~/.astcache/tools.json` per-tool overrides. Agents only see tools from `tools/list`; they cannot request a tier. See [README — Tool tiers](README.md#tool-tiers-and-per-tool-overrides).

## Skills (Ready-Made Instruction Blocks)

**Cursor:** discoverable project skills in [`.cursor/skills/`](.cursor/skills/) (`ast-context-cache-usage`, `ast-context-cache-install`, `ast-context-cache-rebuild`, `ast-context-cache-operator`). See [skills/README.md](skills/README.md).

**Other editors:** portable copy-paste in `skills/` — [agents](skills/agents/SKILL.md), [install](skills/install/SKILL.md), [usage](skills/usage/SKILL.md), [operator](skills/operator/SKILL.md).

# Available Tools

## Core
- **get_context_capsule** - BM25+vector hybrid search with modes: full, skeleton, summary, auto
- **search_semantic** - Natural language search; optional `doc_type` filter (`code`, `doc`, etc.)
- **get_file_context** - All symbols in a file; **default mode `skeleton`** (use instead of reading files)
- **get_project_map** - Project structure overview (~200 tokens at depth=2)
- **get_impact_graph** - Blast radius of a symbol before making changes
- **index_status** - Check if a project is indexed
- **search_docs** - Search cached library/framework documentation
- **list_doc_sources** - List tracked documentation sources (core, read-only)
- **retrieve** - RAG-style retrieval: hybrid search + reranking + context assembly (code + docs)

## Extended
- **index_files** - Index a file or directory (starts file watcher)
- **cache_summary** - Cache summaries for future queries
- **analyze_dead_code** - Find unused functions, classes, imports
- **analyze_complexity** - Find hard-to-maintain code by cyclomatic complexity
- **export_bundle** / **import_bundle** - Portable code bundles without re-indexing
- **fetch_doc** - Fetch, cache, and return external documentation (prefer over WebFetch)
- **add_doc_source** / **remove_doc_source** / **update_doc_source** - Track and manage cached doc URLs

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
- Pass the same `session_id` on **get_context_capsule**, **search_semantic**, **retrieve**, and **get_file_context**
- Prevents re-sending symbols already returned in this conversation; dedup adds to **`dedup_tokens_saved`** in responses and the dashboard

### Token savings tracking
- **Formula:** `tokens_saved = max(0, full_source_baseline − tokens_returned) + dedup_skips`
- **Tracked tools:** `get_context_capsule`, `get_file_context`, `search_semantic`, `retrieve` (each response includes `tokens_saved`, `tokens_used`, `symbol_baseline_tokens`)
- **Not tracked:** doc tools (`fetch_doc`, `search_docs`), indexing (`index_*`), maps/graphs — dashboard **Tokens saved** stays 0 on doc-only days
- **`mode=full`** saves ~nothing; prefer **`auto`** or **`skeleton`** for measurable savings

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

### 10. Use search_docs and fetch_doc for Documentation
- Try **search_docs** first for library/framework docs (local FTS cache)
- On cache miss, use **fetch_doc** (not WebFetch) so URLs are stored in ast-context-cache
- Tracked sources re-fetch when older than **7 days**; use `update_doc_source` or `fetch_doc` with `force_refresh` sooner

### Recommended Workflow
1. index_status → index_files if needed
2. get_project_map (depth=2)
3. get_context_capsule with mode='auto' and session_id
4. get_file_context with default mode='skeleton' (not full) unless implementing
5. search_semantic for intent; retrieve for RAG bundles
6. cache_summary on key symbols; use mode='summary' later
7. get_impact_graph before changing exports
8. search_docs for library/framework questions; fetch_doc when not cached

**Dashboard (operators):** http://localhost:7830 — embed queue gauge, project filter, MCP vs indexing in Recent, tool performance (CPU/latency). See [skills/operator/SKILL.md](skills/operator/SKILL.md).

### Optional search filters (get_context_capsule, search_semantic, retrieve)

- **path_prefix** — limit to symbols under a project-relative path (e.g. `internal/mcp`) or absolute prefix.
- **language** — `go`, `python`, `typescript`, `javascript`, `rust`, `yaml`, etc. (file extension filter).
- **doc_type** — on `search_semantic` only: e.g. `code`, `doc`.
- **kinds** / **kind** — comma-separated or single symbol kind (`function`, `method`, …).

### Pipeline observability

- **get_context_capsule** — response includes **pipeline**: `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse`.
- **retrieve** — **stats** adds the same hybrid counts plus `after_dedup`, `chunks_in_budget`, `tokens_est_all_chunks`, and stage timings (`code_retrieve_ms`, `docs_retrieve_ms`, `dedup_budget_ms`, `search_time_ms`).
