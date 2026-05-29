# ast-context-cache Usage Skill

## When to Use

When you need to search, understand, or analyze code in a project efficiently.
Use this skill when the user asks to:
- Find functions, classes, or symbols
- Understand code structure
- Find dependencies or impact of changes
- Search code by meaning/intent
- Find unused or complex code
- Search documentation for libraries/frameworks
- Share indexed code with another machine (bundles)
- Run code analysis against search results

**Not for server config** (embeddings, dashboard settings, log retention) — use [operator/SKILL.md](../operator/SKILL.md).

## Agent workflow (follow in order)

```
1. index_status(project_path="/absolute/path/to/repo")
2. index_files(path="...", project_path="...")     # if needed; starts watcher + embed queue
3. get_project_map(project_path="...", depth=2)    # ~200 tokens orientation
4. get_context_capsule(query="...", project_path="...", mode="auto", session_id="...")
5. get_file_context(file="...", project_path="...", mode="skeleton")  # one file; default skeleton
6. search_semantic(query="...", project_path="...", session_id="...")  # intent / exploration
7. get_impact_graph(symbol="...", project_path="...")                   # before edits
8. retrieve(query="...", project_path="...", session_id="...")          # RAG code + docs
```

Generate a stable **`session_id`** per conversation (e.g. UUID) and pass it on **`get_context_capsule`**, **`search_semantic`**, **`retrieve`**, and **`get_file_context`** so symbols already returned are not sent again.

## Quick Reference

| Step | Tool | Key params |
|------|------|------------|
| Indexed? | `index_status` | `project_path` absolute |
| Index | `index_files` | `path`, `project_path`; extended tier |
| Orient | `get_project_map` | `depth=2` |
| Keyword search | `get_context_capsule` | `mode=auto`, `session_id`, `token_budget` |
| One file | `get_file_context` | **`mode=skeleton`** (default), `session_id`, `token_budget` |
| By meaning | `search_semantic` | `session_id`, `token_budget`, optional `doc_type` |
| Blast radius | `get_impact_graph` | `symbol`, `project_path` |
| RAG | `retrieve` | `session_id`, `token_budget`, `format` |
| Library docs | `search_docs` | FTS over cached sources |

## Tool availability (tiers)

If `index_files`, `execute_code`, or other tools are missing from `tools/list`, the server tier is too low or the tool is disabled in `~/.astcache/tools.json`. Agents cannot change tier—ask the user to set `AST_MCP_TIER` / edit overrides and restart ast-mcp. See [README](../../README.md#tool-tiers-and-per-tool-overrides).

## Tool selection

| Need | Tool | Notes |
|------|------|-------|
| Find specific function/class | `get_context_capsule` | `mode=auto` (default) |
| Explore unfamiliar code | `get_context_capsule` or `search_semantic` | `mode=skeleton` |
| All symbols in one file | `get_file_context` | **`mode=skeleton` default**; use `full` only when implementing |
| Search by meaning | `search_semantic` | Natural language; optional `doc_type` |
| Project tree | `get_project_map` | `depth=2` (~200 tokens) |
| Before changing exports | `get_impact_graph` | Shows dependents |
| Dead code | `analyze_dead_code` | Extended tier |
| Complexity hotspots | `analyze_complexity` | Extended tier |
| Cache findings | `cache_summary` | Enables `mode=summary` later |
| Best single-shot context | `retrieve` | Code + optional docs, reranked |
| External library docs | `search_docs` then `fetch_doc` | Not WebFetch when MCP is up |
| List doc URLs | `list_doc_sources` | Core tier |
| Portable index | `export_bundle` / `import_bundle` | Extended tier |
| Transform results in JS | `execute_code` | Complete tier; `DATA` variable |

## Mode defaults (important)

| Tool | Default mode | Agent should use |
|------|--------------|------------------|
| `get_context_capsule` | `auto` | Keep `auto` for most searches |
| `get_file_context` | **`skeleton`** | Keep `skeleton` to review a file; `full` only when editing |
| `search_semantic` | `skeleton` | Signatures unless you need full bodies |
| `retrieve` | skeleton-style chunks | `include_source=true` for full source |

| Mode | Token use | When |
|------|-----------|------|
| `auto` | ~20% of full | `get_context_capsule` — full for top hits, skeleton for rest |
| `skeleton` | ~10% of full | Exploration, file review |
| `summary` | ~6% of full | After `cache_summary` for those symbols |
| `full` | 100% | Implementation details only |

## Token savings tracking

`tokens_saved = max(0, full_source_baseline − tokens_returned) + session_dedup_skips`

| Counted | Not counted |
|---------|-------------|
| `get_context_capsule`, `get_file_context`, `search_semantic`, `retrieve` | `fetch_doc`, `search_docs`, `index_*`, maps, impact graph, … |

- Context tool JSON includes **`tokens_saved`**, **`tokens_used`**, **`symbol_baseline_tokens`**, **`dedup_tokens_saved`**
- Dashboard **Tokens saved** = sum of those MCP calls only (doc-only days → **0** is normal)
- **`mode=full`** ≈ no savings; **`auto`** / **`skeleton`** / **`summary`** are where savings come from
- Same **`session_id`** on all four context tools for dedup credit

## RAG: `retrieve`

```
retrieve(
  query="how does authentication work",
  project_path="/absolute/path/to/repo",
  session_id="conv-uuid",
  token_budget=4000,
  format="markdown"
)
```

- `include_docs=false` — code only  
- `include_source=false` (default) — signatures/skeleton in code chunks; `true` for full source  
- `mode` — `skeleton`, `auto`, or `full` when `include_source` is false  
- Filters: `path_prefix`, `language`, `kinds` / `kind` (same as search tools)

Response **`stats`**: hybrid counts (`bm25_candidates`, `vector_candidates`, …), `after_dedup`, `chunks_in_budget`, timings, plus **`tokens_saved`**, **`symbol_baseline_tokens`**, **`dedup_tokens_saved`**, **`budget_tokens_saved`**.

## Optional search filters

On **`get_context_capsule`**, **`search_semantic`**, and **`retrieve`**:

| Parameter | Purpose |
|-----------|---------|
| `path_prefix` | Subtree only (e.g. `internal/mcp`) |
| `language` | `go`, `python`, `typescript`, `javascript`, `yaml`, … |
| `kinds` / `kind` | `function`, `method`, … |
| `doc_type` | **`search_semantic` only** — e.g. `code`, `doc` |

## Pipeline observability

- **`get_context_capsule`** → `pipeline`: `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse`
- **`retrieve`** → `stats` with hybrid counts and `code_retrieve_ms`, `docs_retrieve_ms`, etc.

## Documentation (Context7-style)

```
search_docs(query="useState hook", limit=5)
fetch_doc(name="React", type="markdown", url="https://...", version="18")
list_doc_sources()
update_doc_source(id=1)
remove_doc_source(id=1)
```

Tracked sources re-fetch when older than **7 days** (daily background check). Types: `markdown`, `html`, `json`.

## Indexing notes for agents

- **`index_files`** starts an **fsnotify** watcher and queues **embeddings** (priority + background channels). Large repos fill the queue gradually—use **`index_status`** and dashboard embed gauges if the user cares about progress.
- **Plain `.log` / `.txt`** are not indexed unless enabled in dashboard Settings (FTS only, no embeddings).
- **Watcher ignore globs** in Settings skip noisy paths (e.g. `dist/**`, `*.pb.go`).
- **`file_watcher`** events are logged for observability—they are **not** MCP tools and do not appear in the Tool Usage chart.

## Supported languages

Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish, YAML (use `language` filter by extension).

## When MCP is unavailable

Use grep only for exact symbols in known files. For exploration or cross-module intent, ask the user to start ast-mcp (`make run` or `ast-mcp start`) then use MCP.

## Ports

| Service | URL |
|---------|-----|
| MCP | `http://localhost:7821/mcp` |
| Health | `http://localhost:7821/health` |
| Dashboard | `http://localhost:7830` (operators — see [operator/SKILL.md](../operator/SKILL.md)) |

`project_path` must be the **absolute** repository root.
