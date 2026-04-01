# ast-context-cache

A local-first AST context engine for AI coding agents. Indexes your codebase into a SQLite database using tree-sitter, then serves precise, token-efficient search results over MCP. No cloud, no account, no data leaves your machine.

## Features

- **Multi-language** -- Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish, YAML
- **BM25-ranked full-text search** -- FTS5 virtual table with fallback LIKE scoring
- **Semantic embeddings** -- Optional ONNX-based vector search for semantic similarity
- **Dependency graph** -- Import/call edges stored in an `edges` table; query blast radius with `get_impact_graph`
- **Source code in results** -- Search results include the actual source, not just file pointers
- **File watcher** -- `fsnotify`-based incremental re-indexing with debounce
- **Dashboard** -- Web UI on port 7830 with stats, charts, and recent query history
- **WAL mode** -- SQLite write-ahead logging + busy timeout for concurrency

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

## Configure Your Editor

### Cursor

Add to your MCP settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
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
8. search_docs  →  search cached library/framework documentation
```

### Token Optimization

| Mode | Token Usage | Best For |
|------|-------------|----------|
| `auto` | ~20% | Most searches (default) |
| `skeleton` | ~10% | Exploring unfamiliar code |
| `summary` | ~6% | High-level overviews (requires `cache_summary` first) |
| `full` | 100% | Complete implementation details |

Always pass `session_id` to avoid re-sending symbols already seen in the conversation.

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
