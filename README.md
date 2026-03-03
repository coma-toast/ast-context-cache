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

## MCP Tools

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory using tree-sitter AST parsing. Extracts symbols and import edges. |
| `index_status` | Get statistics about indexed symbols (total files, symbol count). |
| `get_context_capsule` | BM25-ranked full-text search across indexed symbols. Returns source code. |
| `search_semantic` | Semantic search by meaning using vector embeddings. |
| `get_file_context` | Get all symbols in a specific file with mode-aware output. |
| `get_project_map` | Project structure overview at configurable depth. |
| `get_impact_graph` | Find the blast radius of a symbol -- files that import or depend on it. |
| `cache_summary` | Store a summary for a file/symbol for cheap future lookups. |
| `reset_project` | Delete indexed data for a specific project. |
| `reset_all` | Delete all indexed data for all projects. |
| `sync_remote` | Push/pull vectors to remote Milvus for cross-machine sync. |

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
