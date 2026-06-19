# Agent Code Research Guidelines

## Running the MCP server

Start the server from this repo with `make run`, or use `ast-mcp start` after `make install`. **Optional:** For an external launcher that supervises `ast-mcp`, registers Cursor/OpenCode/Claude, and manages `~/.astcache/tools.json`, see [mcp-local](https://github.com/coma-toast/mcp-local) — agent workflows: [mcp-local/AGENTS.md](https://github.com/coma-toast/mcp-local/blob/main/AGENTS.md).

## Tool tiers (server policy)

The host sets which MCP tools exist via **`AST_MCP_TIER`** (`core` | `extended` | `complete`) and optional **`~/.astcache/tools.json`** per-tool `enabled` / `tier` overrides (`AST_MCP_TOOLS_CONFIG` for path). **`AST_MCP_CODE_MODE=false`** disables `execute_code`. Config is read at **ast-mcp startup** only.

Agents see only `tools/list` results—they cannot negotiate tier over MCP. If `index_files` or `execute_code` is missing, ask the user to raise tier or adjust `tools.json` and restart. Full tables and examples: [README.md](README.md#tool-tiers-and-per-tool-overrides), [skills/tools.json.example](skills/tools.json.example).

## Skills / Ready-Made Agent Instructions

### Cursor (discoverable)

Project skills under [`.cursor/skills/`](.cursor/skills/) load when the task matches their description:

| Skill | When it applies |
|-------|-----------------|
| `ast-context-cache-usage` | MCP search, RAG, modes, filters, **virtual context compaction** |
| `ast-context-cache-install` | Install, MCP config, tool tiers, **virtual context tier requirements** |
| `ast-context-cache-rebuild` | Rebuild or restart ast-mcp after code changes |
| `ast-context-cache-operator` | Embeddings, dashboard settings, logs, **virtual context limits** |

See [skills/README.md](skills/README.md) for syncing from portable sources.

### Other editors (portable)

The `skills/` directory has copy-paste blocks and MCP JSON:

| Skill | Contents |
|-------|----------|
| `skills/agents/SKILL.md` | MCP config for OpenCode, Cursor, Claude, VS Code, JetBrains |
| `skills/install/SKILL.md` | Install, troubleshoot, **tool tiers for virtual context** |
| `skills/usage/SKILL.md` | Tool selection, RAG, token tips, **virtual context compaction** |
| `skills/operator/SKILL.md` | Embeddings, dashboard, log retention |

Read `skills/<name>/SKILL.md` when not using Cursor project skills.

## Standard agent workflow

Use MCP in this order for unfamiliar code (generate a stable **`session_id`** per conversation):

1. **`index_status`** — `project_path` must be the **absolute** repo root (for WTG worktrees, use the checkout path e.g. `~/spaces/nightly/slapi`, not the main git clone).
2. **`index_files`** — if unindexed (extended tier); starts watcher + embed queue.
3. **`get_project_map`** — `depth=2` (~200 tokens).
4. **`get_context_capsule`** — `mode=auto` (default), `session_id`, `token_budget` (default 4000).
5. **`get_file_context`** — **`mode=skeleton`** (default); use `full` only when implementing.
6. **`search_semantic`** — natural-language / intent; pass `session_id`, `token_budget`.
7. **`get_impact_graph`** — before changing exported symbols.
8. **`retrieve`** — single-shot RAG (code ± docs); `session_id`, filters as needed.

**Virtual context (long threads):** Before host compaction, **`store_context`** (extended) with the same `session_id`; keep `ctx_*` stubs in chat. After compaction, **`fetch_context`** or **`search_context`** (core). **`flush_context`** when done.

**Defaults:** `get_context_capsule` → `auto`; `get_file_context` → **`skeleton`**; `search_semantic` → `skeleton`. Do not read whole source files when MCP can return structured symbols.

**Operators** (embeddings, dashboard, log retention): [skills/operator/SKILL.md](skills/operator/SKILL.md) — dashboard http://localhost:7830 (embed queue gauge; tool performance with CPU/latency).

## MCP Server Tools (Preferred)

When working with codebases that have an MCP server available, **always prefer MCP tools** over direct grep/read/glob:

- **get_context_capsule** - Search code with token-efficient modes (auto, skeleton, summary)
- **search_semantic** - Natural language search: "function that handles auth"
- **get_file_context** - Get all symbols in a specific file (use instead of reading files)
- **get_impact_graph** - See all files depending on a symbol before making changes
- **retrieve** - RAG-style retrieval combining code + docs into formatted context
- **search_docs** - Search cached library/framework documentation
- **cache_summary** - Cache your own summaries for future queries

### All Tools

#### Core

| Tool | Description |
|------|-------------|
| `get_context_capsule` | BM25+vector hybrid search. Modes: `full`, `skeleton`, `summary`, `auto`. |
| `search_semantic` | Semantic search by meaning using vector embeddings. Optional `doc_type` (e.g. `code`, `doc`). |
| `get_file_context` | All symbols in a file; **default `skeleton`**. Use instead of reading files; pass `session_id`, `token_budget`. |
| `get_project_map` | Project structure overview (depth 1=dirs, 2=files, 3=symbols). |
| `get_impact_graph` | Blast radius of a symbol -- files that import or depend on it. |
| `index_status` | Check if a project is indexed. Returns file/symbol counts. |
| `search_docs` | Search locally cached documentation (FTS). Try before WebFetch for library/framework docs. |
| `list_doc_sources` | List all tracked documentation sources (read-only). |
| `retrieve` | RAG-style retrieval: hybrid search + reranking + context assembly (code + docs). Supports `markdown`, `xml`, `json` output. |
| `fetch_context` | Retrieve offloaded virtual context by `ctx_*` ref(s). |
| `list_context` | List stored virtual context refs for a session (metadata only). |
| `search_context` | Find stored virtual context by keyword/meaning. |

#### Extended

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory. Starts a file watcher for incremental re-indexing. Plain `.log` is not indexed unless enabled in dashboard settings; watcher ignore globs apply to paths that would otherwise be indexed as code. |
| `cache_summary` | Store a summary for a file/symbol for cheap future lookups. |
| `store_context` | Offload arbitrary conversation/code notes before compaction; returns stable `ctx_*` refs. |
| `flush_context` | Delete stored virtual context (session, refs, or all). |
| `analyze_dead_code` | Find unused functions, classes, and imports. |
| `analyze_complexity` | Calculate cyclomatic complexity to find hard-to-maintain code. |
| `export_bundle` | Export indexed code as a portable `.astbundle` file. |
| `import_bundle` | Import a previously exported bundle without re-indexing. |
| `fetch_doc` | Fetch a documentation URL, store it in the local cache, and return entries (prefer over WebFetch). |
| `add_doc_source` | Track a documentation URL for async background caching (markdown, html, webpage, json). |
| `remove_doc_source` | Remove a tracked documentation source. |
| `update_doc_source` | Manually refresh a documentation source. |

#### Complete

| Tool | Description |
|------|-------------|
| `execute_code` | Run JavaScript in a sandbox against search results. Only output enters context. |

### Mode Selection

| Mode | Use Case | Token Savings |
|------|----------|---------------|
| `auto` | **`get_context_capsule` default** — full for top hits, skeleton for rest | ~80% |
| `skeleton` | **`get_file_context` / `search_semantic` default** — exploration | ~90% |
| `summary` | High-level overviews (requires cache_summary first) | ~94% |
| `full` | Only when you need complete implementation details | 0% |

### Best Practices

1. **Use session_id** - On `get_context_capsule`, `search_semantic`, `retrieve`, and `get_file_context` to skip symbols already returned
2. **Set token_budget** - Default 4000, adjust based on need
3. **Use get_project_map first** - ~200 tokens for full project overview
4. **Use get_file_context over read** - Default **`skeleton`**; only use `full` when editing implementation
5. **Cache summaries** - Call cache_summary after understanding key files
6. **Use search_docs** - For library/framework documentation; use **`fetch_doc`** (not WebFetch) when the cache misses
7. **Optional filters** - On `get_context_capsule`, `search_semantic`, and `retrieve`, pass `path_prefix`, `language`, `kinds` / `kind`, and on `search_semantic` optional `doc_type` to narrow results
8. **Supported languages** - Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish, YAML (see README for full list)
9. **Pipeline stats** - `get_context_capsule` returns `pipeline` counts; `retrieve` stats include hybrid-stage counts and timings (see README / CLAUDE.md)
10. **Indexing load** - Embeddings go through a bounded **queue** with workers (dashboard: embed queue / active). **Pin** heavy projects in Settings for priority embedding, no idle watcher stop, and warmer vector unload behavior

### Virtual context compaction

Host editors drop bulky chat history when the context window fills. Virtual context keeps that material on disk with stable refs so agents can recover it without re-pasting megabytes into chat.

| Tool | Tier | Role |
|------|------|------|
| `store_context` | extended | Offload text → `ctx_*` ref + quota stats |
| `fetch_context` | core | Recover by ref(s); primary post-compaction path |
| `list_context` | core | List refs/labels for a session (no full content) |
| `search_context` | core | Keyword + hybrid search when refs lost from chat |
| `flush_context` | extended | Delete notes; free quota; stubs become invalid |

**Why:** Preserve reasoning, plans, long diffs, and summaries across compaction without burning context window on repeat content.

**When to store:** Long analysis you will reference later; before compaction warnings; when the thread is getting large. Pass the same **`session_id`** as `get_context_capsule` / `retrieve`.

**When to fetch vs search:** Use **`fetch_context`** when chat still has `ctx_*` stubs. Use **`list_context`** to discover refs. Use **`search_context`** when stubs were summarized away.

**When to flush:** Conversation finished; `context_limit_exceeded` errors; user asks to clear stored notes. Operators can also flush all via dashboard Settings.

**Parameters:**

```
store_context(content, session_id, label?, project_path?, tags?)
fetch_context(refs, session_id?)     # session_id enforces ownership
list_context(session_id, project_path?, limit?)
search_context(query, session_id?, project_path?, limit?)
flush_context(session_id? | refs? | all=true, project_path?)
```

**Limits (defaults):** 50 notes / 32k tokens per session; 500 notes / 200k tokens global. Policy `reject` or `lru_session` for session caps (global always rejects). Configure in dashboard **Settings → Virtual context** or env `AST_CONTEXT_MAX_*`.

**Metrics (separate from code Tokens saved):**

| Event | Tracked as |
|-------|------------|
| `store_context` | `virtual_tokens_stored`; query log `tokens_saved` |
| `fetch_context` / `search_context` | `virtual_tokens_returned`; query log `tokens_used` |
| `flush_context` | freed tokens in flush stats |

Dashboard **Virtual context** card: active inventory, 30d stored vs accessed, utilization %, orphan notes (stored but never fetched), flushed tokens. API: `GET http://localhost:7830/api/context-stats`.

**Chat pattern:** After store, write `[ctx_…] label` in the thread instead of the full content. After compaction, fetch by ref.

### Token savings tracking

Savings are measured vs **full source** for symbols actually returned:

`tokens_saved = max(0, symbol_baseline − tokens_returned) + dedup_skips`

| Tools that increment dashboard **Tokens saved** | Tools that do not |
|-------------------------------------------------|-------------------|
| `get_context_capsule`, `get_file_context`, `search_semantic`, `retrieve` | `fetch_doc`, `search_docs`, `index_*`, `get_project_map`, `get_impact_graph`, … |

- Responses include **`tokens_saved`**, **`tokens_used`**, **`symbol_baseline_tokens`**, optional **`dedup_tokens_saved`**
- **`mode=full`** → ~0 savings; use **`auto`** / **`skeleton`** for real reductions
- Pass **`session_id`** on all four context tools for dedup credit
- Dashboard sublabel on **Tokens saved** shows 30d total, **avg/day**, **dedup**, and **vs files**

### Documentation Tools

Track and search external documentation (similar to Context7):

```
search_docs(query="useState hook", limit=5)
fetch_doc(name="React", type="markdown", url="https://...", version="18")
add_doc_source(name="React", type="markdown", url="https://...", version="18")  # async track only
list_doc_sources()
update_doc_source(id=1)
remove_doc_source(id=1)
```

**Workflow:** `search_docs` first → on miss, **`fetch_doc`** (registers + fetches in one call). Do not use WebFetch for library docs when MCP is available.

Tracked sources re-fetch automatically when older than **7 days** (daily background check). Use `update_doc_source` or `fetch_doc` with `force_refresh=true` to refresh sooner. Types: `markdown`, `html`, `webpage` (JS-rendered via Playwright Firefox), `json`. On `fetch_doc`, pass `render_js=true` to store as `webpage`.

## When MCP Not Available

Use grep/ast_grep only when:
- You know exactly what to search (single keyword)
- Known file location
- MCP server is not running

Avoid:
- Multi-angle searches (use MCP search_semantic instead)
- Cross-module pattern discovery
- Unfamiliar codebase exploration

## Slide / `slidehq` workspaces

When researching or changing code in Slide repositories (for example `common`, `sandbox`, `box`, `cloud`, `agent`), **prefer shared logic and patterns from [`github.com/slidehq/common`](https://github.com/slidehq/common)** over inventing new abstractions in a single repo. Check common for existing utilities, errors, task patterns, retries, and logging before adding parallel implementations elsewhere.
