# ast-context-cache

A local-first AST context engine for AI coding agents. Indexes your codebase into a SQLite database using tree-sitter, then serves precise, token-efficient search results over MCP. No cloud, no account, no data leaves your machine.

## Features

- **Multi-language** -- Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish, YAML
- **BM25-ranked full-text search** -- FTS5 virtual table with fallback LIKE scoring
- **Semantic embeddings** -- Default ONNX (all-mpnet, 768-d) or pluggable Ollama / HTTP / OpenAI-compatible remote (must stay 768-d to match the index)
- **Dependency graph** -- Import/call edges stored in an `edges` table; query blast radius with `get_impact_graph`
- **Source code in results** -- Search results include the actual source, not just file pointers
- **File watcher** -- `fsnotify`-based incremental re-indexing with debounce; dashboard **ignore globs** skip high-churn code paths after `IsCodeFile`
- **Logs** -- Plain `.log` / `.txt` are not indexed by default; enable in dashboard for FTS-only search (no embeddings). Optional **log retention** deletes only `.log` under absolute roots you configure
- **Dashboard** -- Web UI on port 7830 with stats, charts, and recent query history
- **WAL mode** -- SQLite write-ahead logging + busy timeout for concurrency; **synchronous=NORMAL** and a **32 MiB page cache** reduce fsync pressure; **PASSIVE WAL checkpoints** every 10 minutes (less write amplification than aggressive truncate). Per-file indexing uses **one transaction** per file; MCP **query** and **session** dedup rows are **batched** before insert.

## Quick Start

**Prerequisites:** Go 1.21+ with CGO enabled, and `brew` on macOS (for ONNX Runtime).

```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
make setup
```

That's it. `make setup` will:
1. Install ONNX Runtime (via `brew install onnxruntime` on macOS)
2. Download the embedding model (all-mpnet-base-v2)
3. Download the pre-built tokenizer library for your platform
4. Build the binary

Then run the server:

```bash
make run
```

Output:
```
MCP: http://localhost:7821/mcp
Dashboard: http://localhost:7830
```

## Embedding backends

The vector store is built for **768-dimensional** L2-normalized embeddings. The default is local **ONNX** (no extra services). Alternatives use environment variables and do not require downloading `model.onnx` for the main process (unless you switch back to `onnx`).

| `EMBED_BACKEND` | When to use | Main env vars |
|-----------------|------------|---------------|
| `onnx` (default) | Full local path: `make setup` pulls HuggingFace ONNX + tokenizer | `MODEL_DIR` to override model directory |
| `ollama` | Local or Docker [Ollama](https://ollama.com) with a **768-d** model; default `nomic-embed-text` | `OLLAMA_HOST` (e.g. `http://127.0.0.1:11434`), `OLLAMA_EMBED_MODEL` |
| `http` | Any service that matches the built-in JSON: `POST` body `{"texts":["..."]}` → `{"embeddings":[[float32,...]]}` (same as `http://localhost:7821/embed` on the ONNX server) | `EMBED_HTTP_URL` (default `http://127.0.0.1:8080/embed`), `EMBED_HTTP_BEARER` |
| `openai` (alias: `litellm`) | [LiteLLM](https://docs.litellm.ai/docs/), OpenAI, or any **OpenAI-compatible** `POST /v1/embeddings` gateway; vectors must be **768-d** (native model or `dimensions` in JSON) | `EMBED_OPENAI_BASE_URL` (default `https://api.openai.com/v1`), **`EMBED_OPENAI_MODEL`** (required), `EMBED_OPENAI_API_KEY`, `EMBED_OPENAI_DIMENSIONS` (optional: unset sends `768` for v3 shortening; `0` omits the field) |

**OpenAI / LiteLLM:** Point `EMBED_OPENAI_BASE_URL` at your gateway root including `/v1` (e.g. `https://your-litellm/v1`). If you change embedding model or dimensionality, clear or re-index so stored vectors stay consistent with the 768-d index.

**Docker (Ollama only):** from the repo root, `docker compose -f docker/compose.ollama-embed.yml up -d`, then `docker exec -it ollama-embed ollama pull nomic-embed-text`, then run ast-mcp with `EMBED_BACKEND=ollama`.

**Process environment:** Whatever starts `ast-mcp` (foreground terminal, the `ast-mcp` shell function from `make install`, systemd, or another supervisor) must have the embedding variables from the table above exported for non-default backends—for example set `EMBED_BACKEND=ollama` and `OLLAMA_HOST`, or `EMBED_BACKEND=openai`, `EMBED_OPENAI_BASE_URL`, `EMBED_OPENAI_API_KEY`, and `EMBED_OPENAI_MODEL`, in the same environment as the process that execs `./ast-mcp`.

**Dashboard (easier):** On **Settings** (port 7830), use **Embedding backend** to save the same keys into local SQLite (`~/.astcache/usage.db`). **Non-empty environment variables always override** the saved values. **Restart ast-mcp** after changing embedding settings so `NewForMain` runs again.

`GET /health` and `GET /embed/health` include `embed_mode`, `embed_model`, and `backend` so you can confirm which path is active.

## Shell Function (optional)

Install a shell function for easy start/stop management:

```bash
make install
```

This installs an `ast-mcp` function into your shell config (fish, bash, and/or zsh — whichever it finds).

```bash
ast-mcp start      # start the server (background, logs to /tmp/ast-mcp.log)
ast-mcp stop       # stop the server
ast-mcp restart    # restart
ast-mcp status     # show running status + URLs
ast-mcp health     # hit the /health endpoint
ast-mcp log        # tail the log file
ast-mcp build      # rebuild the binary
ast-mcp dash       # open the dashboard in a browser
```

To uninstall: `make uninstall`

## Optional: mcp-local launcher

This repository ships only the **`ast-mcp`** binary. If you prefer a separate Go CLI to start it, merge editor MCP JSON with other servers, and manage tool tiers, use **[mcp-local](https://github.com/coma-toast/mcp-local)** (its own repository). Clone and build from that project, then follow **[AGENTS.md](https://github.com/coma-toast/mcp-local/blob/main/AGENTS.md)** (agents) and **README** for `~/.mcp-local/config.yaml`, `tools sync` / `tools apply`, and `json tools`—details are not duplicated here so they stay in sync with that tool.

To manage per-tool enable/tier overrides from a launcher, use mcp-local (or any tool) to write `~/.astcache/tools.json` via the same schema as below, then **restart ast-mcp**.

## Tool tiers and per-tool overrides

The MCP server can expose a **subset** of tools. Tiers are cumulative: `complete` includes `extended` and `core`; `extended` includes `core`.

| Tier | Typical tools |
|------|----------------|
| **core** | Search, status, maps, docs, `retrieve` (read-only) |
| **extended** | core + `index_files`, `cache_summary`, bundles, doc sources, analysis |
| **complete** | extended + `execute_code` (requires code mode) |

**Global controls (environment):**

| Variable | Values | Default |
|----------|--------|---------|
| `AST_MCP_TIER` | `core`, `extended`, `complete` | `complete` |
| `AST_MCP_CODE_MODE` | `false` / `0` disables `execute_code` | enabled |
| `AST_MCP_TOOLS_CONFIG` | Path to JSON overrides file | `~/.astcache/tools.json` |

Example — read-only agent profile:

```bash
AST_MCP_TIER=core make run
```

Example — indexing without sandbox execution:

```bash
AST_MCP_TIER=extended AST_MCP_CODE_MODE=false make run
```

**Per-tool overrides** (`~/.astcache/tools.json`, or path from `AST_MCP_TOOLS_CONFIG`):

```json
{
  "execute_code": { "enabled": false, "tier": "complete" },
  "index_files": { "enabled": true, "tier": "core", "description": "Indexing allowed in core profile" }
}
```

- **`enabled`**: `false` removes the tool from `tools/list` and rejects calls.
- **`tier`**: Effective minimum tier for that tool (can promote an extended tool to `core`, or keep defaults when omitted).
- **`description`**: Optional replacement text in `tools/list` when set.

Overrides are loaded **at process start**; restart `ast-mcp` after editing the file. See [`skills/tools.json.example`](skills/tools.json.example).

**With [mcp-local](https://github.com/coma-toast/mcp-local):** `mcp-local tools sync ast-context-cache` (server must be running) → `mcp-local tools` TUI to enable/disable tools and set tiers → `mcp-local restart ast-context-cache` (writes `tools.json`, sets `AST_MCP_TIER` / `AST_MCP_CODE_MODE` / `AST_MCP_TOOLS_CONFIG`). Or `mcp-local tools apply ast-context-cache` then restart. In Go, `mcp.SaveToolConfigs` writes the same file schema.

Agents **cannot** request a tier over MCP; they only see tools the server exposes. If a call fails, the response explains disabled vs tier vs code mode.

## Configure Your Editor

### Cursor

Add to your MCP settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp",
      "env": {
        "AST_MCP_TIER": "extended",
        "AST_MCP_CODE_MODE": "false"
      }
    }
  }
}
```

### OpenCode

Add to your `opencode.jsonc`:

```jsonc
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

## For AI Agents

Instructions for AI coding agents to install, run, and recover the MCP server.

### Install & Run

```bash
cd /path/to/ast-context-cache
make setup   # one-time: installs deps, downloads model, builds binary
make run     # starts server on :7821 (MCP) and :7830 (dashboard)
```

If the shell function is installed (`make install`), use `ast-mcp start` instead.

### Verify It's Running

```bash
curl -s http://localhost:7821/health
```

Expected response: `{"service":"ast-context-cache","status":"healthy","version":"1.0.0"}`

### Crash Recovery

The server may crash or stop unexpectedly. Follow these steps to recover:

1. **Check if the process is alive:**
   ```bash
   pgrep -f ast-mcp || echo "not running"
   ```

2. **If not running, restart it:**
   ```bash
   cd /path/to/ast-context-cache && make run
   ```
   Or with the shell function: `ast-mcp restart`

3. **If the port is stuck (address already in use):**
   ```bash
   lsof -i :7821          # find the stale process
   kill <pid>             # kill it
   make run               # restart
   ```

4. **If the binary is broken or stale:**
   ```bash
   cd /path/to/ast-context-cache
   make build && make run
   ```

5. **Health check after restart:**
   ```bash
   curl -s http://localhost:7821/health
   ```

### Recommended Workflow

```
1. index_status  →  check if project is indexed
2. index_files   →  index if needed (starts file watcher)
3. get_project_map depth=2  →  orient yourself (~200 tokens)
4. get_context_capsule mode=auto  →  search code (top 3 full, rest skeleton)
5. get_file_context  →  all symbols in a specific file
6. get_impact_graph  →  blast radius before modifying a symbol
7. cache_summary  →  save what you learned for future queries
8. retrieve  →  RAG-style retrieval (code + docs → formatted context)
9. search_docs  →  search cached library/framework documentation
```

### Token Optimization

| Mode | Token Usage | Best For |
|------|-------------|----------|
| `auto` | ~20% | Most searches (default) |
| `skeleton` | ~10% | Exploring unfamiliar code |
| `summary` | ~6% | High-level overviews (requires `cache_summary` first) |
| `full` | 100% | Complete implementation details |

Always pass `session_id` to avoid re-sending symbols already seen in the conversation.

### Optional search filters

For **`get_context_capsule`**, **`search_semantic`**, and **`retrieve`**, you can narrow results before hybrid ranking:

| Parameter | Purpose |
|-----------|---------|
| `path_prefix` | Only symbols under this path (project-relative, e.g. `internal/mcp`, or an absolute path prefix). |
| `language` | Coarse language filter: `go`, `python`, `typescript`, `javascript`, `rust`, etc. (uses file extensions). |
| `kinds` | Comma-separated symbol kinds (e.g. `function,method`). |
| `kind` | Single kind (same as one entry in `kinds`). |

All are optional; omitting them preserves previous behavior.

When filters are set, **BM25 (FTS) and fallback SQL** apply `kind`, `language` (file suffix), and `path_prefix` constraints in the query where possible, so fewer rows are scanned before ranking. A final Go-side check still enforces the same rules for edge cases.

### Pipeline observability

- **`get_context_capsule`** responses include a `pipeline` object: `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse` (counts after filters, through BM25 + vector + RRF stages).
- **`retrieve`** `stats` include those counts plus `after_dedup`, `chunks_in_budget`, `tokens_est_all_chunks`, and timing fields (`code_retrieve_ms`, `docs_retrieve_ms`, `dedup_budget_ms`, `search_time_ms`).

### Indexing queue, pinning, and warm vectors

- **Embedding work** runs through a **bounded queue** with multiple workers so rapid file changes do not spawn unbounded ONNX goroutines. The dashboard **Index health** section shows **embed queue** depth and **active** embedding workers.
- **Pinned projects** (Settings → **Pin** on a project): file-change embeddings are **prioritized** on the high queue; **watchers are not auto-stopped** for idle timeout on pinned projects; **vector cache idle unload** uses a **longer effective timeout** when any project is pinned (warmer “tier” without a separate cold store).

## MCP Tools

### Core

| Tool | Description |
|------|-------------|
| `get_context_capsule` | BM25+vector hybrid search across indexed symbols. Modes: `full`, `skeleton`, `summary`, `auto`. |
| `search_semantic` | Semantic search by meaning using vector embeddings. |
| `get_file_context` | Get all symbols in a specific file with mode-aware output. Use instead of reading files directly. |
| `get_project_map` | Project structure overview at configurable depth (1=dirs, 2=files, 3=symbols). |
| `get_impact_graph` | Find the blast radius of a symbol -- files that import or depend on it. |
| `index_status` | Check if a project is indexed. Returns file/symbol counts. |
| `search_docs` | Search locally cached documentation by title or content (FTS). |
| `retrieve` | RAG-style retrieval: hybrid search + reranking + context assembly. Returns formatted context ready for LLM. |

### Extended

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory using tree-sitter AST parsing. Starts a file watcher. |
| `cache_summary` | Store a summary for a file/symbol for cheap future lookups. |
| `analyze_dead_code` | Find unused functions, classes, and imports. |
| `analyze_complexity` | Calculate cyclomatic complexity to find hard-to-maintain code. |
| `export_bundle` | Export indexed code as a portable `.astbundle` file. |
| `import_bundle` | Import a previously exported bundle without re-indexing. |
| `add_doc_source` | Add a documentation URL to track and cache (markdown, html, json). |
| `remove_doc_source` | Remove a tracked documentation source. |
| `list_doc_sources` | List all tracked documentation sources. |
| `update_doc_source` | Manually refresh a documentation source. |

### Complete

| Tool | Description |
|------|-------------|
| `execute_code` | Run JavaScript code in a sandbox against search results. Only output enters context. |

## Architecture

```
┌─────────────┐    JSON-RPC 2.0    ┌──────────────────┐
│  AI Agent    │ ◄───────────────► │  MCP Server :7821 │
│  (Cursor,    │                   │                    │
│   OpenCode)  │                   │  tree-sitter AST   │
└─────────────┘                   │  SQLite + FTS5     │
                                   │  ONNX Embeddings   │
                                   │  fsnotify watcher  │
                                   └──────────────────┘
                                           │
                                   ┌───────┴────────┐
                                   │ Dashboard :7830 │
                                   │ (self-contained │
                                   │  HTML + JS)     │
                                   └──────────────────┘
```

## Database Schema

- **symbols** -- `name`, `kind`, `file`, `start_line`, `end_line`, `code`, `fqn`, `project_path`
- **symbols_fts** -- FTS5 virtual table over `name`, `fqn`, `code`
- **vectors** -- Vector embeddings for semantic search (if model loaded)
- **edges** -- `source_file`, `source_symbol`, `target`, `kind`, `project_path`
- **queries** -- Tool call log with timestamps, durations, token savings
- **summaries** -- Cached summaries for files/symbols
- **doc_sources** -- Tracked documentation URLs with type, version, content hash
- **doc_content** -- Cached documentation entries (title, content, section)
- **docs_fts** -- FTS5 virtual table over `doc_content` for full-text doc search

## Configuration

The database is stored at `~/.astcache/usage.db`. The dashboard frontend is served from a `dist/` directory relative to the binary.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNXRUNTIME_LIB` | Path to ONNX Runtime library | Auto-detected from brew/system |
| `MODEL_DIR` | Path to model files | `./model` (next to binary) |
| `DB_PATH` | Path to SQLite database | `~/.astcache/usage.db` |

## Cross-platform Build

The Makefile's `download-tokenizer-lib` target picks a pre-built `libtokenizers.a` from [daulet/tokenizers releases](https://github.com/daulet/tokenizers/releases) based on `GOOS` and `GOARCH` (e.g. darwin-arm64, darwin-amd64, linux-amd64, linux-arm64). If your platform is not supported, the Makefile will report it and you can download the matching tarball manually and extract `libtokenizers.a` into the repo root.

## Linux

On Linux, install ONNX Runtime before running `make setup`:

```bash
# Ubuntu/Debian
apt-get install libonnxruntime-dev

# Or download from https://github.com/microsoft/onnxruntime/releases
# and place libonnxruntime.so in /usr/lib/ or /usr/local/lib/
```

Then `make setup` will detect it and proceed normally.

## License

MIT
