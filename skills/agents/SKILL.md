# Agent Integration Configuration

## What This File Is

How to wire ast-context-cache into an editor or agent host, plus a pasteable instruction block for project `AGENTS.md` / `CLAUDE.md`. Prefer the installer: it writes the MCP entry, these skills, and the rule or instruction block for each host, merge-only and with backups.

### Cursor project skills (this repo)

When working **in this repository**, prefer discoverable skills under [`.cursor/skills/`](../../.cursor/skills/):

- `ast-context-cache-usage` — MCP search, RAG, modes, filters, **virtual context compaction**, **subagent handoff**
- `ast-context-cache-install` — setup, installer, tiers (**extended** for store_context)
- `ast-context-cache-rebuild` — rebuild/restart after server changes
- `ast-context-cache-operator` — embeddings, dashboard, **virtual context limits**

Portable sources live in `skills/`; sync notes in [skills/README.md](../README.md).

---

## Install with the installer (recommended)

```bash
ast-mcp install --target <host> --dry-run   # preview the per-file diff
ast-mcp install --target <host> --yes       # apply (backs up every file first)
ast-mcp verify                              # status per host and component
```

`ast-mcp` is the shell function from `make install` (it passes these subcommands to the binary); `./ast-mcp` in the repo works the same.

Hosts: `claude_code`, `cursor`, `opencode`, `codex`, `claude_desktop`, `vscode`, `jetbrains`, or `all`. Components: `mcp`, `skills`, `rules`, `hooks` (Claude Code subagent-handoff hooks: available, opt-in behind the `feature_handoff_hooks` flag; see [docs/handoff.md](../../docs/handoff.md#claude-code-hooks)). The dashboard offers the same under Settings → **Agent integration**. Reference: [docs/INSTALL.md](../../docs/INSTALL.md#connect-your-agents); exact files and keys per host: [docs/host-integration.md](../../docs/host-integration.md).

---

## Editor MCP Configuration (manual)

These are the entries the installer writes. URL entries take **no `env` block** (it does nothing for an HTTP server); set `AST_MCP_TIER` and similar on the `ast-mcp` process.

### Claude Code
File: `~/.claude.json` (top level), or `claude mcp add --transport http --scope user ast-context-cache http://127.0.0.1:7821/mcp`
```json
{
  "mcpServers": {
    "ast-context-cache": { "type": "http", "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

### Cursor
File: `~/.cursor/mcp.json`
```json
{
  "mcpServers": {
    "ast-context-cache": { "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

### OpenCode
File: `~/.config/opencode/opencode.jsonc` (or `opencode.json`). The key is `mcp`, not `mcpServers`.
```jsonc
{
  "mcp": {
    "ast-context-cache": { "type": "remote", "url": "http://127.0.0.1:7821/mcp", "enabled": true }
  }
}
```

### Codex
File: `~/.codex/config.toml`
```toml
[mcp_servers.ast-context-cache]
url = "http://127.0.0.1:7821/mcp"
```

### VS Code (GitHub Copilot)
File: user `mcp.json` (run "MCP: Open User Configuration")
```json
{
  "servers": {
    "ast-context-cache": { "type": "http", "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

### Claude Desktop
File: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `~/.config/Claude/claude_desktop_config.json` (Linux). Claude Desktop launches stdio servers only, so it needs a bridge: [mcp-local](https://github.com/coma-toast/mcp-local)'s `bridge` command (use its absolute path), or `npx -y mcp-remote`.
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "command": "/absolute/path/to/mcp-local",
      "args": ["bridge", "http://127.0.0.1:7821/mcp"]
    }
  }
}
```

### JetBrains (AI Assistant)
No config file: Settings → Tools → AI Assistant → Model Context Protocol (MCP) → Add, paste the JSON below, set the server level to Global, then OK and Apply.
```json
{ "mcpServers": { "ast-context-cache": { "url": "http://127.0.0.1:7821/mcp" } } }
```

---

## Tool tiers (host configuration)

The MCP server does not let agents pick a tier. The **operator** sets:

- `AST_MCP_TIER=core|extended|complete` on the ast-mcp process
- `AST_MCP_CODE_MODE=false` to hide `execute_code`
- Optional `~/.astcache/tools.json` (or `AST_MCP_TOOLS_CONFIG`) for per-tool `enabled` / `tier` / `description`
- Feature flags (dashboard Settings → Features, or `AST_FEATURE_*` env locks) to switch features such as handoff on and off live

Restart ast-mcp after changing `tools.json`. Example file: [`skills/tools.json.example`](../tools.json.example). See [README tool tiers](../../README.md#tool-tiers-and-per-tool-overrides).

---

## Agent Instructions Block (paste into AGENTS.md or CLAUDE.md)

The installer writes the short canonical block from [`instructions/agents-block.md`](../../instructions/agents-block.md) into each host's global instructions file. For a project `AGENTS.md` / `CLAUDE.md` that wants the full tool surface, paste this longer block:

```markdown
# MCP Code Search (ast-context-cache)

**Goals:** Token-efficient local code search (tree-sitter + hybrid BM25/vector), cached library docs, impact analysis, and **virtual context compaction** so long threads survive host chat compaction.

MCP server: http://127.0.0.1:7821/mcp · Dashboard: http://127.0.0.1:7830

When working with this codebase, **always prefer MCP tools** over direct grep/read/glob.

## Quick Workflow

1. `index_status` — check if project is indexed
2. `index_files` — index if needed (starts file watcher). Large directories return a `job_id` with `status: running`; poll `index_status` (`index_jobs`, `indexing`) until `completed`
3. `get_project_map depth=2` — orient yourself (~200 tokens)
4. `get_context_capsule mode=auto` + `session_id` — search code (top hits full, rest skeleton)
5. `get_file_context mode=skeleton` — all symbols in a file (default skeleton, not full)
6. `get_impact_graph` — blast radius before modifying a symbol
7. `cache_summary` — save what you learned for future queries
8. `retrieve` — RAG-style retrieval (code + docs in one call)
9. `search_docs` — search cached library/framework documentation; `no_match: true` means nothing relevant is cached → `fetch_doc`
10. `store_context` — offload bulky thread text before host compaction (extended); keep `ctx_*` stubs
11. `fetch_context` / `search_context` — recover offloaded notes after compaction (core)
12. `edit_context` — keep a stored note current or shrink it **in place**, same `ctx_*` ref (extended); manage context as a file rather than re-storing it, and avoid compaction to make room
13. `open_handoff` — **first call** when your prompt contains `[handoff hof_…]`; then use the child `session_id` it returns
14. `handoff` — `create` to delegate (paste the stub into the subagent prompt), `complete` to finish as a child, `collect` / `list` to fan in or recover refs

## Core Tools

| Tool | Description |
|------|-------------|
| `get_context_capsule` | BM25+vector hybrid search. Modes: `full`, `skeleton`, `summary`, `auto`. |
| `search_semantic` | Semantic search by meaning using vector embeddings. Optional `doc_type`. |
| `get_file_context` | All symbols in a file; **default `skeleton`**. Pass `session_id`. Use instead of reading files directly. |
| `get_project_map` | Project structure overview (depth 1=dirs, 2=files, 3=symbols). |
| `get_impact_graph` | Blast radius of a symbol — files that import or depend on it. |
| `diff_impact` | Blast radius of a branch (`base_ref`...`head_ref`) or a GitHub PR (`pr`). |
| `check_symbol_exists` | Whether a name is declared anywhere, with every declaring file and line. |
| `check_deletion_safety` | Symbols a change removes: still referenced (unsafe) vs no callers (safe). |
| `index_status` | Check if a project is indexed. Returns file/symbol counts. |
| `search_docs` | Search locally cached documentation (FTS). Try before WebFetch. |
| `list_doc_sources` | List all tracked documentation sources (read-only). |
| `retrieve` | RAG-style retrieval: hybrid search + reranking + context assembly (code + docs). |
| `fetch_context` | Retrieve offloaded virtual context by `ctx_*` ref(s). Primary recovery after compaction. |
| `list_context` | List stored virtual context refs for a session (metadata only). |
| `search_context` | Find stored virtual context by keyword/meaning when refs are lost. |
| `recall_memory` | Compact structured facts/procedures (`mem_*`). |
| `handoff` | Delegate and fan in: `create`, `complete`, `collect`, `list`, `status`, `flush`. |
| `open_handoff` | Child entry point: `open` (before any search), `expand`, `resume`. |
| `scratchpad` | Shared findings, dead ends, and advisory claims for parallel subagents: `post`, `read`, `retract`, `claim`, `release`. |

## Extended Tools

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory. Starts a file watcher for incremental re-indexing. |
| `cache_summary` | Store a summary for a file/symbol for cheap future lookups. |
| `store_context` | Offload conversation/code notes before compaction; returns stable `ctx_*` refs. |
| `edit_context` | Edit a stored note in place, keeping its `ctx_*` ref (`append`/`replace`/`delete`/`rewrite`/`revert`); reports `tokens_reclaimed`. |
| `define_context_fn` / `apply_context_fn` | Define a named reusable transform and re-invoke it across notes or a session. Default off. |
| `flush_context` | Delete stored virtual context (session, refs, or all). Frees quota. |
| `store_memory` / `forget_memory` | Write or invalidate structured memory (`mem_*`). |
| `analyze_dead_code` | Find unused functions, classes, and imports. |
| `analyze_complexity` | Calculate cyclomatic complexity to find hard-to-maintain code. |
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

## Virtual context compaction (required when tier allows)

**Do not rely on host compaction alone.** Use ast-context-cache to offload bulky thread text before the editor drops it.

| Tool | Tier | When |
|------|------|------|
| `store_context` | extended | Before compaction; plans, diffs, long analysis (~70% context) |
| `fetch_context` | core | Recover by `ctx_*` ref after compaction |
| `list_context` | core | List refs/labels for a session |
| `search_context` | core | Refs lost from chat; search by topic |
| `edit_context` | extended | Any change to stored content — edit in place, same ref, reversible |
| `flush_context` | extended | Thread done or quota exceeded |

**Manage context as a file.** Once a note exists, keep it current with `edit_context` rather than flushing and re-storing: a re-store invalidates the stub already in chat, mints a new ref, and debits quota again. Watch `tokens_reclaimed` to confirm shrinking paid off, use `revert` instead of starting over, and avoid compaction as a way to make room.

Use the **same `session_id`** as code search. Chat stub: `[ctx_a1b2c3d4e5f6] label`.

Metrics: dashboard **Virtual context** card (separate from code **Tokens saved**). Requires **`AST_MCP_TIER=extended`** for store/flush.

## Subagent handoff

- Prompt contains `[handoff hof_…]` → `open_handoff(handoff=…)` before any search; pass the returned child `session_id` everywhere; finish with `handoff(action=complete, content, status, summary)` and output the returned stub.
- Delegating → `handoff(action=create, session_id, brief, pointers)` → paste the stub into the subagent prompt. `mode=fork` only for a host fork that inherited your window.
- Parallel subagents share one stub and coordinate through `scratchpad` (claims are advisory). Fan in with `handoff(action=collect)`; after compaction, `handoff(action=list)` recovers lost refs.
- Never put credentials in a brief or scratchpad post.

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

## Cursor Rules (global or project)

The canonical always-apply rule is [`rules/cursor/ast-context-cache.mdc`](../../rules/cursor/ast-context-cache.mdc) (`alwaysApply: true`): session ids, efficient exploration, host compaction, and subagent handoffs. `ast-mcp install --target cursor --component rules --yes` writes it to `~/.cursor/rules/ast-context-cache.mdc` between version-stamped markers. Cursor documents global rules as Settings → Rules (User Rules) only, so if the global file is not picked up, paste the rule body there instead.

**Project optional:** copy the same file to `.cursor/rules/ast-context-cache.mdc`.
