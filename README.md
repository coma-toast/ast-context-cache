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
go build -tags sqlite_fts5 -o ast-mcp .

# Run
./ast-mcp
# MCP server: http://localhost:7821/mcp
# Dashboard:  http://localhost:7830
```

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

## License

MIT
