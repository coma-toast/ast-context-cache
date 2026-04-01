# Installing ast-context-cache

## Prerequisites

- Go 1.25+
- macOS (with Homebrew) or Linux
- onnxruntime library

## Quick Install

```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
make setup
```

This installs all dependencies and builds the binary.

## Manual Steps

### 1. Install Dependencies
```bash
make deps
```
This installs onnxruntime (via brew on macOS), downloads the embedding model, and downloads the tokenizer library.

### 2. Build
```bash
make build
```

### 3. Run
```bash
make run
```
Starts the MCP server on `http://localhost:7821/mcp`

## Shell Function (Optional)

```bash
make install
```
Adds `ast-mcp start|stop|restart|status|health` shell function to your shell config.

## MCP Server Configuration

### OpenCode
Add to `~/.config/opencode/opencode.jsonc`:
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### Cursor
Add to `.cursor/mcp.json` in your project:
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
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

### Claude Code
Add to `CLAUDE.md` in your project:
```markdown
# Code Context Instructions

Use ast-context-cache MCP server (http://localhost:7821/mcp) for efficient code search.
- get_context_capsule: Search code with token-efficient modes
- cache_summary: Cache your own summaries
- analyze_dead_code: Find unused code
```

## Dashboard

Visit `http://localhost:7821/` to see:
- Query statistics and token savings
- Index health metrics
- Recent queries with toast notifications
- System resource usage (RAM/disk)
- Project management (reset/delete)

## Troubleshooting

### "library 'tokenizers' not found"
```bash
make download-tokenizer-lib
```

### Model files missing
```bash
make download-model
```

### Port already in use
```bash
lsof -i :7821
# Kill the existing process or change the port
```
