---
name: ast-context-cache-install
description: Use when installing, configuring, or troubleshooting ast-mcp, MCP editor config, tool tiers (AST_MCP_TIER), or tools.json overrides.
---

# ast-context-cache Installation

## When to Use

When you need to install or setup ast-context-cache for an AI coding agent.
Use this skill when the user asks to:
- Install ast-context-cache
- Setup MCP server for their agent
- Configure Cursor, OpenCode, Claude, etc.
- Troubleshoot installation issues

## Installation Steps

### 1. Clone and Setup
```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
make setup
```

### 2. Start Server
```bash
make run
```
Server runs at `http://localhost:7821/mcp`
Dashboard at `http://localhost:7830`

### 3. Configure Your Agent

#### OpenCode
Add to `~/.config/opencode/opencode.jsonc`:
```jsonc
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

#### Cursor
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

Discoverable Cursor skills in this repo: `.cursor/skills/ast-{usage,install,rebuild,operator}/`.

#### Claude Desktop
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

## Tool tiers

Host sets `AST_MCP_TIER=core|extended|complete` and optional `~/.astcache/tools.json`. Restart ast-mcp after changes. Example: [skills/tools.json.example](../../skills/tools.json.example).

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
```

## Shell Function (Optional)
```bash
make install
```
Adds `ast-mcp start|stop|restart|status|health` to your shell.

Canonical copy: [skills/install/SKILL.md](../../skills/install/SKILL.md)
