# Code Context Instructions

## Goals

**ast-context-cache** is a local-first AST context engine: index code with tree-sitter, search over MCP with minimal tokens, cache docs, and **offload conversation context before host compaction** so agents recover plans/analysis after the editor compacts chat.

| Priority | Mechanism |
|----------|-----------|
| Token-efficient code | `get_context_capsule` `mode=auto`, `get_file_context` `skeleton`, session dedup, `execute_code` + `code_script_hints` |
| Accurate edits | `get_impact_graph`, `retrieve`, optional filters (`path_prefix`, `language`, `kinds`) |
| Long threads | **`store_context`** → `[ctx_…]` stubs → **`fetch_context`** / **`search_context`** (same `session_id`) |

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

Read `skills/<name>/SKILL.md` when not using Cursor project skills.

### Virtual context compaction (agents) — required when tier allows

**Do not rely on host compaction alone** for bulky analysis, plans, or diffs you will need later. Use ast-context-cache virtual context:

1. **`store_context`** (extended) with the same **`session_id`** as code search → keep **`ctx_*` stub** in chat.
2. After compaction → **`fetch_context`**, **`list_context`**, or **`search_context`** (core).
3. When done → **`flush_context`**.

Full guide: [skills/usage/SKILL.md](skills/usage/SKILL.md#virtual-context-compaction).

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
- **fetch_context** - Retrieve offloaded virtual context by `ctx_*` ref(s) after host compaction
- **list_context** - List stored virtual context refs for a session (metadata only)
- **search_context** - Find stored virtual context by keyword/meaning when refs are lost

## Extended
- **index_files** - Index a file or directory (starts file watcher)
- **cache_summary** - Cache summaries for future queries
- **store_context** - Offload conversation/code notes with stable `ctx_*` refs before compaction
- **flush_context** - Delete stored virtual context to free quota
- **analyze_dead_code** - Find unused functions, classes, imports
- **analyze_complexity** - Find hard-to-maintain code by cyclomatic complexity
- **export_bundle** / **import_bundle** - Portable code bundles without re-indexing
- **fetch_doc** - Fetch, cache, and return external documentation (prefer over WebFetch)
- **add_doc_source** / **remove_doc_source** / **update_doc_source** - Track and manage cached doc URLs

## Complete
- **execute_code** - Run JS sandbox on search JSON; optional `script_id`; `tokens_saved` in response

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
- **Tracked tools:** `get_context_capsule`, `get_file_context`, `search_semantic`, `retrieve`, `execute_code` (each response includes `tokens_saved`, `tokens_used`, `symbol_baseline_tokens` or `data_baseline_tokens`)
- **Not tracked:** doc tools (`fetch_doc`, `search_docs`), indexing (`index_*`), maps/graphs — dashboard **Tokens saved** stays 0 on doc-only days
- **`mode=full`** saves ~nothing; prefer **`auto`** or **`skeleton`** for measurable savings
- **`execute_code`:** `tokens_saved = max(0, data_baseline_tokens − tokens_used)` when shrinking search JSON via scripts

### Code-mode scripts (agents)

Search tools may return **`code_script_hints`**. If non-empty: `execute_code(script_id=..., data=<results JSON>, project_path=...)` and use **`result` only**. Requires complete tier + `AST_MCP_CODE_MODE`. Repo scripts: `{project}/scripts/code-mode/` — [scripts/code-mode/README.md](scripts/code-mode/README.md).

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

### 11. Virtual context compaction (conversation offload)

**Why:** When the host compacts chat, long analysis and plans vanish from the model window. **store_context** saves that text locally and returns a short **`ctx_*` ref** you keep in chat instead of the full body.

**When to store:** Bulky thread content you may need later; before compaction or ~70% context fill. Requires **extended** tier (`store_context`).

**When to recover:** After compaction — **`fetch_context(refs=[...])`** if you kept stubs; **`list_context`** to see what's stored; **`search_context(query=...)`** if refs were lost. Read tools are **core** tier.

**When to flush:** Thread done or quota exceeded — **`flush_context(session_id=...)`** (extended). Dashboard **Virtual context** card tracks inventory vs access (separate from code **Tokens saved**).

Use the **same `session_id`** as code search tools. Example chat stub: `[ctx_a1b2c3d4e5f6] auth design`.

### Recommended Workflow
1. index_status → index_files if needed
2. get_project_map (depth=2)
3. get_context_capsule with mode='auto' and session_id
4. get_file_context with default mode='skeleton' (not full) unless implementing
5. search_semantic for intent; retrieve for RAG bundles
6. cache_summary on key symbols; use mode='summary' later
7. get_impact_graph before changing exports
8. search_docs for library/framework questions; fetch_doc when not cached
9. store_context before host compaction — keep ctx_* stubs; fetch_context after compaction

**Dashboard (operators):** http://localhost:7830 — embed queue gauge, project filter, MCP vs indexing in Recent, tool performance (CPU/latency). See [skills/operator/SKILL.md](skills/operator/SKILL.md).

### Optional search filters (get_context_capsule, search_semantic, retrieve)

- **path_prefix** — limit to symbols under a project-relative path (e.g. `internal/mcp`) or absolute prefix.
- **language** — `go`, `python`, `typescript`, `javascript`, `rust`, `yaml`, etc. (file extension filter).
- **doc_type** — on `search_semantic` only: e.g. `code`, `doc`.
- **kinds** / **kind** — comma-separated or single symbol kind (`function`, `method`, …).

### Pipeline observability

- **get_context_capsule** — response includes **pipeline**: `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse`.
- **retrieve** — **stats** adds the same hybrid counts plus `after_dedup`, `chunks_in_budget`, `tokens_est_all_chunks`, and stage timings (`code_retrieve_ms`, `docs_retrieve_ms`, `dedup_budget_ms`, `search_time_ms`).
