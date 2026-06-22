# Agent Integration Configuration

## What This File Is

Copy-paste-ready MCP config snippets and agent instruction blocks for integrating
ast-context-cache into your editor and projects. Pick the section for your editor,
then copy the `AGENTS.md` / `CLAUDE.md` block into your project root.

### Cursor project skills (this repo)

When working **in this repository**, prefer discoverable skills under [`.cursor/skills/`](../.cursor/skills/):

- `ast-context-cache-usage` — MCP search, RAG, modes, filters, **virtual context compaction**
- `ast-context-cache-install` — setup, tiers (**extended** for store_context)
- `ast-context-cache-rebuild` — rebuild/restart after server changes
- `ast-context-cache-operator` — embeddings, dashboard, **virtual context limits**

Portable sources live in `skills/`; sync notes in [skills/README.md](../README.md).

---

## Editor MCP Configuration

### OpenCode
File: `~/.config/opencode/opencode.jsonc`
```jsonc
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### Cursor
File: `.cursor/mcp.json`
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

Project skills: `.cursor/skills/ast-{usage,install,rebuild,operator}/` (see [skills/README.md](../README.md)).

### VS Code (GitHub Copilot)
File: `.vscode/mcp.json`
```json
{
  "servers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### Claude Desktop
File: `~/Library/Application Support/Claude/claude_desktop_config.json`
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "command": "http",
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### JetBrains (AI Assistant)
File: `.idea/mcp.json`
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

---

## Tool tiers (host configuration)

The MCP server does not let agents pick a tier. The **operator** sets:

- `AST_MCP_TIER=core|extended|complete` on the ast-mcp process
- `AST_MCP_CODE_MODE=false` to hide `execute_code`
- Optional `~/.astcache/tools.json` (or `AST_MCP_TOOLS_CONFIG`) for per-tool `enabled` / `tier` / `description`

Restart ast-mcp after changing `tools.json`. Example file: [`skills/tools.json.example`](../tools.json.example). See [README tool tiers](../../README.md#tool-tiers-and-per-tool-overrides).

---

## Agent Instructions Block (paste into AGENTS.md or CLAUDE.md)

```markdown
# MCP Code Search (ast-context-cache)

When working with this codebase, **always prefer MCP tools** over direct grep/read/glob.
MCP server: http://localhost:7821/mcp

## Quick Workflow

1. `index_status` — check if project is indexed
2. `index_files` — index if needed (starts file watcher)
3. `get_project_map depth=2` — orient yourself (~200 tokens)
4. `get_context_capsule mode=auto` + `session_id` — search code (top hits full, rest skeleton)
5. `get_file_context mode=skeleton` — all symbols in a file (default skeleton, not full)
6. `get_impact_graph` — blast radius before modifying a symbol
7. `cache_summary` — save what you learned for future queries
8. `retrieve` — RAG-style retrieval (code + docs in one call)
9. `search_docs` — search cached library/framework documentation
10. `store_context` — offload bulky thread text before host compaction (extended); keep `ctx_*` stubs
11. `fetch_context` / `search_context` — recover offloaded notes after compaction (core)

## Core Tools

| Tool | Description |
|------|-------------|
| `get_context_capsule` | BM25+vector hybrid search. Modes: `full`, `skeleton`, `summary`, `auto`. |
| `search_semantic` | Semantic search by meaning using vector embeddings. Optional `doc_type`. |
| `get_file_context` | All symbols in a file; **default `skeleton`**. Pass `session_id`. Use instead of reading files directly. |
| `get_project_map` | Project structure overview (depth 1=dirs, 2=files, 3=symbols). |
| `get_impact_graph` | Blast radius of a symbol — files that import or depend on it. |
| `index_status` | Check if a project is indexed. Returns file/symbol counts. |
| `search_docs` | Search locally cached documentation (FTS). Try before WebFetch. |
| `list_doc_sources` | List all tracked documentation sources (read-only). |
| `retrieve` | RAG-style retrieval: hybrid search + reranking + context assembly (code + docs). |
| `fetch_context` | Retrieve offloaded virtual context by `ctx_*` ref(s). Primary recovery after compaction. |
| `list_context` | List stored virtual context refs for a session (metadata only). |
| `search_context` | Find stored virtual context by keyword/meaning when refs are lost. |

## Extended Tools

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory. Starts a file watcher for incremental re-indexing. |
| `cache_summary` | Store a summary for a file/symbol for cheap future lookups. |
| `store_context` | Offload conversation/code notes before compaction; returns stable `ctx_*` refs. |
| `flush_context` | Delete stored virtual context (session, refs, or all). Frees quota. |
| `analyze_dead_code` | Find unused functions, classes, and imports. |
| `analyze_complexity` | Calculate cyclomatic complexity to find hard-to-maintain code. |
| `export_bundle` | Export indexed code as a portable `.astbundle` file. |
| `import_bundle` | Import a previously exported bundle without re-indexing. |
| `fetch_doc` | Fetch a doc URL, cache it, and return stored entries (prefer over WebFetch). |
| `add_doc_source` | Track a doc URL for async background caching. |
| `remove_doc_source` | Remove a tracked documentation source. |
| `update_doc_source` | Manually refresh a documentation source. |

## Complete Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Run JS on search JSON (`data`); optional `script_id`; `tokens_saved` in response. Requires complete tier + `AST_MCP_CODE_MODE`. |

### Code-mode scripts

Search tools may return **`code_script_hints`**. Workflow: hint `script_id` → `execute_code(script_id, data=<results JSON>, project_path=...)` → use `result` only. Repo scripts: `{project}/scripts/code-mode/` — [scripts/code-mode/README.md](../../scripts/code-mode/README.md).

## Mode Selection

| Mode | Use Case | Token Savings |
|------|----------|---------------|
| `auto` | **`get_context_capsule` default** — full for top hits, skeleton for rest | ~80% |
| `skeleton` | **`get_file_context` / `search_semantic` default** | ~90% |
| `summary` | High-level overviews (requires cache_summary first) | ~94% |
| `full` | Only when you need complete implementation details | 0% |

## Best Practices

1. **Use session_id** — on get_context_capsule, search_semantic, retrieve, get_file_context
2. **Set token_budget** — default 4000; adjust based on need
3. **Use get_project_map first** — ~200 tokens for full project overview
4. **Use get_file_context over read** — structured, mode-aware output
5. **Cache summaries** — call cache_summary after understanding key files
6. **Use search_docs** — for library/framework docs; use **fetch_doc** (not WebFetch) when not cached
7. **Optional filters** — `path_prefix`, `language`, `kinds`/`kind` on get_context_capsule, search_semantic, retrieve
8. **Pipeline stats** — get_context_capsule returns `pipeline` counts; retrieve `stats` includes timing + budget info
9. **Pinned projects** — pin in Settings for priority embedding, no idle watcher stop, warmer vector cache

## Virtual context compaction

**Why:** Host chat compaction drops long analysis from the model window. Store locally, keep short **`ctx_*` stubs** in chat, fetch after compaction.

| Tool | Tier | When |
|------|------|------|
| `store_context` | extended | Before compaction; bulky plans/diffs/analysis |
| `fetch_context` | core | Recover by ref after compaction |
| `list_context` | core | Discover stored refs (metadata only) |
| `search_context` | core | Refs lost from chat; search by topic |
| `flush_context` | extended | Thread done or over quota |

Use the **same `session_id`** as code search. Example stub: `[ctx_a1b2c3d4e5f6] label`.

Metrics are on the dashboard **Virtual context** card (not code **Tokens saved**). Requires **`AST_MCP_TIER=extended`** for store/flush.

## Token savings tracking

- **Formula:** `tokens_saved = max(0, full_source_baseline − tokens_returned) + dedup_skips`
- **Tracked:** `get_context_capsule`, `get_file_context`, `search_semantic`, `retrieve`, `execute_code` (fields in JSON: `tokens_saved`, `tokens_used`, `symbol_baseline_tokens` or `data_baseline_tokens`, `dedup_tokens_saved`)
- **Not tracked:** `fetch_doc`, `search_docs`, `index_*`, etc. — dashboard **Tokens saved** can be 0 on doc-only days
- Use **`auto`** / **`skeleton`**; **`mode=full`** saves ~nothing. Pass **`session_id`** on all four context tools.

## Optional Search Filters

For `get_context_capsule`, `search_semantic`, and `retrieve`:

| Parameter | Purpose |
|-----------|---------|
| `path_prefix` | Only symbols under this path (e.g. `internal/mcp`) |
| `language` | Filter by language: `go`, `python`, `typescript`, `javascript`, `yaml`, etc. |
| `kinds` | Comma-separated symbol kinds (e.g. `function,method`) |
| `kind` | Single kind filter |
| `doc_type` | On `search_semantic` only: e.g. `code`, `doc` |

## Documentation Tools

```
search_docs(query="useState hook", limit=5)
fetch_doc(name="React", type="markdown", url="https://...", version="18")
list_doc_sources()
update_doc_source(id=1)
remove_doc_source(id=1)
```

Tracked sources re-fetch when older than **7 days**. Use `fetch_doc` with `force_refresh=true` or `update_doc_source` to refresh sooner. Types: `markdown`, `html`, `json`.

## When MCP Is Not Available

Use grep/read only when:
- You know exactly what to search (single keyword, known file)
- MCP server is confirmed not running

Avoid using grep/read for:
- Multi-angle searches (use search_semantic)
- Cross-module pattern discovery
- Unfamiliar codebase exploration
```

---

## Cursor Rules (optional)

File: `.cursor/rules/ast-context-cache.mdc`
```markdown
---
description: Use ast-context-cache for efficient code search
---

When searching code, prefer the ast-context-cache MCP tools:
- Use get_context_capsule with mode='auto' for most searches
- Use search_semantic for natural language queries
- Use get_impact_graph before making changes to understand blast radius
- Use get_file_context to get all symbols in a specific file
- Use retrieve for RAG-style context assembly (code + docs in one call)
- Always pass session_id for deduplication
- Use search_docs for library/framework documentation
- Before host compaction: store_context with same session_id; keep ctx_* stubs in chat
- After compaction: fetch_context(refs) or search_context(query); flush_context when done
```
