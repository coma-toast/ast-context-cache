# ast-context-cache

A local-first AST context engine for AI coding agents. Indexes your codebase into a SQLite database using tree-sitter, then serves precise, token-efficient search results over MCP. No cloud, no account, no data leaves your machine.

## Features

- **Multi-language** -- Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish
- **BM25-ranked full-text search** -- FTS5 virtual table with fallback LIKE scoring
- **Semantic embeddings** -- Optional ONNX-based vector search for semantic similarity
- **Dependency graph** -- Import/call edges stored in an `edges` table; query blast radius with `get_impact_graph`
- **Source code in results** -- Search results include the actual source, not just file pointers
- **File watcher** -- `fsnotify`-based incremental re-indexing with debounce
- **Dashboard** -- Web UI on port 7830 with stats, charts, and recent query history
- **WAL mode** -- SQLite write-ahead logging + busy timeout for concurrency

## Prerequisites

- **Go 1.21+** with CGO enabled
- **ONNX Runtime** (for embeddings)
- **SQLite** with FTS5 extension
- **tree-sitter** grammars (included)

### Installing ONNX Runtime

**macOS (Intel):**
```bash
brew install onnxruntime
```

**macOS (Apple Silicon):**
```bash
brew install onnxruntime
# or
brew install onnxruntime --prefix=/opt/homebrew
```

**Linux:**
```bash
# Ubuntu/Debian
apt-get install libonnxruntime

# Or download from https://github.com/microsoft/onnxruntime/releases
```

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
```

### 2. Download the embedding model (optional but recommended)

```bash
# Creates model/ directory with all-mpnet-base-v2 ONNX model
./download-model.sh
```

### 3. Download tokenizer library

```bash
# For macOS Intel (x86_64)
curl -L -o libtokenizers.darwin-x86_64.tar.gz \
    https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-x86_64.tar.gz
tar xzf libtokenizers.darwin-x86_64.tar.gz
rm libtokenizers.darwin-x86_64.tar.gz

# For macOS Apple Silicon (arm64)
curl -L -o libtokenizers.darwin-arm64.tar.gz \
    https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-arm64.tar.gz
tar xzf libtokenizers.darwin-arm64.tar.gz
rm libtokenizers.darwin-arm64.tar.gz

# For Linux (x86_64)
curl -L -o libtokenizers.linux-x86_64.tar.gz \
    https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.linux-x86_64.tar.gz
tar xzf libtokenizers.linux-x86_64.tar.gz
rm libtokenizers.linux-x86_64.tar.gz
```

### 4. Build

```bash
# macOS
make build

# Or manually with correct paths
CGO_LDFLAGS="-L$(pwd) -L/usr/local/lib" CGO_CFLAGS="-I/usr/local/include/onnxruntime" \
    go build -tags sqlite_fts5 -o ast-mcp ./cmd/ast-mcp/
```

### 5. Run

```bash
# Set ONNX Runtime library path
export ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.dylib  # macOS Intel
# export ONNXRUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib  # macOS ARM

./ast-mcp
```

Output:
```
Dashboard: http://localhost:7830
MCP: http://localhost:7821/mcp
```

### 6. Configure in OpenCode

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

### 7. Index a project

```bash
curl -s -X POST http://localhost:7821/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "index_files",
      "arguments": {
        "path": "/path/to/your/project",
        "project_path": "/path/to/your/project"
      }
    }
  }'
```

## Fish Shell Setup (Optional)

If you use Fish shell, you can use this function for easy management:

```fish
# Add to ~/.config/fish/functions/mcp-local.fish

function is_port_in_use
    set -l port $argv[1]
    if lsof -iTCP:$port -sTCP:LISTEN -P 2>/dev/null | grep -q .
        return 0
    end
    return 1
end

function check_port_status
    set -l port $argv[1]
    set -l name $argv[2]
    if lsof -iTCP:$port -sTCP:LISTEN -P 2>/dev/null | grep -q .
        set_color green
        echo "  $name: running on port $port"
    else
        set_color red
        echo "  $name: not running (port $port)"
    end
    set_color normal
end

function mcp-local
    if test (count $argv) -eq 0
        echo "Usage: mcp-local <command>"
        echo ""
        echo "Commands:"
        echo "  setup     Install prerequisites and dependencies"
        echo "  start     Start all MCP servers"
        echo "  stop      Stop all MCP servers"
        echo "  restart   Restart all MCP servers"
        echo "  status    Show MCP server status"
        echo "  health    Check health of MCP servers"
        return 0
    end
    
    set -l cmd $argv[1]
    
    switch $cmd
        case start
            echo "Starting MCP servers..."
            # Start ast-context-cache (set ONNXRUNTIME_LIB before running)
            if is_port_in_use 7821
                echo "  ast-context-cache: already running"
            else
                cd /Users/jason/git/ast-context-cache
                set -lx ONNXRUNTIME_LIB /usr/local/lib/libonnxruntime.dylib
                nohup ./ast-mcp > /tmp/ast-mcp.log 2>&1 &
                sleep 2
                if is_port_in_use 7821
                    echo "  ast-context-cache: started"
                else
                    echo "  ast-context-cache: FAILED (check /tmp/ast-mcp.log)"
                end
            end
            echo "Done!"
            
        case setup
            echo "Setting up MCP server prerequisites..."
            set -l ast_dir /Users/jason/git/ast-context-cache
            
            if test -d "$ast_dir"
                cd $ast_dir
                
                # Download model files
                if test -d model
                    echo "  model files: already exist"
                else
                    echo "  Downloading ONNX model..."
                    mkdir -p model
                    curl -L --progress-bar -o model/model.onnx \
                        "https://huggingface.co/onnx-models/all-mpnet-base-v2-onnx/resolve/main/model.onnx"
                    curl -L --progress-bar -o model/tokenizer.json \
                        "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json"
                    echo "  model files: downloaded"
                end
                
                # Download tokenizer library
                set -l arch (uname -m)
                if test -f libtokenizers.a
                    echo "  tokenizer library: already exists"
                else
                    echo "  Downloading tokenizer library for $arch..."
                    if test "$arch" = "arm64"
                        set -l libname libtokenizers.darwin-arm64.tar.gz
                    else
                        set -l libname libtokenizers.darwin-x86_64.tar.gz
                    end
                    curl -L -o $libname "https://github.com/daulet/tokenizers/releases/latest/download/$libname"
                    tar xzf $libname
                    rm -f $libname
                    echo "  tokenizer library: downloaded"
                end
                
                # Build binary
                if test -f ast-mcp
                    echo "  binary: already built"
                else
                    echo "  Building ast-mcp..."
                    CGO_LDFLAGS="-L$ast_dir -L/usr/local/lib" CGO_CFLAGS="-I/usr/local/include/onnxruntime" \
                        go build -tags sqlite_fts5 -o ast-mcp ./cmd/ast-mcp/
                    if test -f ast-mcp
                        echo "  binary: built"
                    else
                        echo "  binary: BUILD FAILED"
                    end
                end
                
                echo "  ast-context-cache: ready"
            else
                echo "  ast-context-cache not found at $ast_dir"
            end
            
        case stop
            set -l ast_pid (lsof -t -iTCP:7821 -sTCP:LISTEN 2>/dev/null)
            if test -n "$ast_pid"
                kill $ast_pid 2>/dev/null
                echo "  ast-context-cache: stopped"
            else
                echo "  ast-context-cache: not running"
            end
            
        case restart
            mcp-local stop
            sleep 1
            mcp-local start
            
        case status
            check_port_status 7821 "ast-context-cache"
            
        case health
            if is_port_in_use 7821
                set -l response (curl -s -m 2 http://localhost:7821/health 2>/dev/null)
                if echo "$response" | grep -q "healthy"
                    set_color green
                    echo "  ast-context-cache: healthy"
                else
                    echo "  ast-context-cache: unknown response"
                end
            else
                set_color red
                echo "  ast-context-cache: not running"
            end
            set_color normal
    end
end
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
- **embeddings** -- Vector embeddings for semantic search (if model loaded)
- **edges** -- `source_file`, `source_symbol`, `target`, `kind`, `project_path`
- **queries** -- Tool call log with timestamps, durations, token savings

## Configuration

The database is stored at `~/.astcache/usage.db`. The dashboard frontend is served from a `dist/` directory relative to the binary.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNXRUNTIME_LIB` | Path to ONNX Runtime library | Required for embeddings |
| `MODEL_DIR` | Path to model files | `./model` (next to binary) |
| `DB_PATH` | Path to SQLite database | `~/.astcache/usage.db` |

## License

MIT
