# ast-context-cache

A local-first AST context engine for AI coding agents. Indexes your codebase into a SQLite database using tree-sitter, then serves precise, token-efficient search results over MCP. No cloud, no account, no data leaves your machine.

## Features

- **Multi-language** -- Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish
- **BM25-ranked full-text search** -- FTS5 virtual table with fallback LIKE scoring
- **Dependency graph** -- Import/call edges stored in an `edges` table; query blast radius with `get_impact_graph`
- **Source code in results** -- Search results include the actual source, not just file pointers
- **File watcher** -- `fsnotify`-based incremental re-indexing with debounce
- **Dashboard** -- Web UI on port 7830 with stats, charts, and recent query history
- **WAL mode** -- SQLite write-ahead logging + busy timeout for concurrency

## Quick Start

```bash
# Build (requires Go 1.21+ and CGO for sqlite3)
# Without embedder: build from repo root
go build -tags sqlite_fts5 -o ast-mcp ./cmd/ast-mcp/

# With embedder (recommended): use Makefile — downloads model + tokenizer lib and builds
make build
```

```bash
# Run (from repo directory so ./model/ is next to the binary)
./ast-mcp
# MCP server: http://localhost:7821/mcp
# Dashboard:  http://localhost:7830
```

## Setup with embedder

For semantic search, hybrid BM25+vector search, and the `/embed` endpoint you need the embedder. On a **new machine** (fresh clone or different box):

**Prerequisites**

- Go 1.25+ with CGO
- ONNX Runtime shared library: macOS `brew install onnxruntime` (e.g. `/opt/homebrew/lib/libonnxruntime.dylib`); Linux install from distro or [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) (default `/usr/lib/libonnxruntime.so`)
- Build via `make` or with equivalent CGO flags

**Steps**

1. Clone the repo and `cd` into it.
2. Run **`make build`**: downloads `model/tokenizer.json` and `model/model.onnx`, downloads pre-built `libtokenizers.a` for your platform, and builds. On non–darwin-arm64 see [Cross-platform build](#cross-platform-build).
3. Run from the **same directory** so the binary sees `./model/`: `./ast-mcp`. If ONNX Runtime is elsewhere: `ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib ./ast-mcp` (or `.so` on Linux).
4. Optional: set **`MODEL_DIR`** to a directory containing `tokenizer.json` and `model.onnx` if the binary is installed elsewhere (e.g. `/usr/local/bin`).

**Troubleshooting**

If the dashboard shows **"Embedder: not loaded"**, check the server log for the real error (e.g. "load tokenizer from ... no such file" or "init ONNX Runtime"). Then verify: (1) `model/tokenizer.json` and `model/model.onnx` exist next to the binary or in `MODEL_DIR`, (2) ONNX Runtime is installed and `ONNXRUNTIME_LIB` points to it if needed, (3) the binary was built with `make build` (or equivalent CGO flags).

## Cross-platform build

The Makefile’s `download-tokenizer-lib` target picks a pre-built `libtokenizers.a` from [daulet/tokenizers releases](https://github.com/daulet/tokenizers/releases) based on `GOOS` and `GOARCH` (e.g. darwin-arm64, darwin-amd64, linux-amd64, linux-arm64). Run `make build`; if your platform is not supported, the Makefile will report it and you can download the matching tarball manually and extract `libtokenizers.a` into the repo root.

## MCP Tools

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory using tree-sitter AST parsing. Extracts symbols and import edges. |
| `index_status` | Get statistics about indexed symbols (total files, symbol count). |
| `get_context_capsule` | BM25-ranked full-text search across indexed symbols. Returns source code. |
| `get_impact_graph` | Find the blast radius of a symbol -- files that import or depend on it. |
| `reset_project` | Delete indexed data for a specific project. |
| `reset_all` | Delete all indexed data for all projects. |

## Architecture

```
┌─────────────┐    JSON-RPC 2.0    ┌──────────────────┐
│  AI Agent    │ ◄───────────────► │  MCP Server :7821 │
│  (Cursor,    │                   │                    │
│   OpenCode)  │                   │  tree-sitter AST   │
└─────────────┘                   │  SQLite + FTS5     │
                                   │  fsnotify watcher  │
                                   └──────────────────┘
                                           │
                                   ┌───────┴────────┐
                                   │ Dashboard :7830 │
                                   │ (self-contained │
                                   │  HTML + JS)     │
                                   └────────────────┘
```

## Database Schema

- **symbols** -- `name`, `kind`, `file`, `start_line`, `end_line`, `code`, `fqn`, `project_path`
- **symbols_fts** -- FTS5 virtual table over `name`, `fqn`, `code`
- **edges** -- `source_file`, `source_symbol`, `target`, `kind`, `project_path`
- **queries** -- Tool call log with timestamps, durations, token savings

## Configuration

The database is stored at `~/.astcache/usage.db`. The dashboard frontend is served from a `dist/` directory relative to the binary.

- **MODEL_DIR** (optional): Directory containing `tokenizer.json` and `model.onnx`. If unset, the binary looks for a `model/` directory next to the executable.
- **ONNXRUNTIME_LIB** (optional): Full path to the ONNX Runtime shared library (`libonnxruntime.dylib` on macOS, `libonnxruntime.so` on Linux). Defaults: `/opt/homebrew/lib/libonnxruntime.dylib` (macOS), `/usr/lib/libonnxruntime.so` (Linux).

## License

MIT
