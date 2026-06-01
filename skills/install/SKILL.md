# ast-context-cache Installation Skill

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
      "url": "http://localhost:7821/mcp",
      "env": {
        "AST_MCP_TIER": "extended"
      }
    }
  }
}
```

**Tier note:** `store_context` and `flush_context` (virtual context compaction) require **extended** tier or higher. Read tools (`fetch_context`, `list_context`, `search_context`) are **core**. Set `AST_MCP_TIER=extended` on the ast-mcp process (or in MCP server env above) and restart after changes.

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

#### Claude Code
Add `CLAUDE.md` to your project:
```markdown
# Code Context Instructions

Use ast-context-cache MCP server (http://localhost:7821/mcp) for efficient code search.
- get_context_capsule: Search code with token-efficient modes
- store_context / fetch_context: Virtual context compaction (extended to store)
- cache_summary: Cache your own summaries
- analyze_dead_code: Find unused code
```

See [usage/SKILL.md](../usage/SKILL.md) for full workflows including virtual context compaction.

## Tool tiers

| Tier | Virtual context |
|------|-----------------|
| core | `fetch_context`, `list_context`, `search_context` |
| extended | + `store_context`, `flush_context` |

Default server tier is `complete`. For read-only agents: `AST_MCP_TIER=core`. For indexing without compaction writes: `AST_MCP_TIER=extended`.

Per-tool overrides: `~/.astcache/tools.json` — see [tools.json.example](../tools.json.example). Restart ast-mcp after edits.

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
