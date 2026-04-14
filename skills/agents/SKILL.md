# Agent Integration Configuration

## OpenCode

### MCP Config
File: `~/.config/opencode/opencode.jsonc`
```jsonc
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### Agent Instructions
Add to project `AGENTS.md` or `CLAUDE.md`:
```markdown
# MCP Server Tools (Preferred)

When working with codebases that have an MCP server available, **always prefer MCP tools** over direct grep/read/glob:

- **get_context_capsule** - Search code with token-efficient modes (auto, skeleton, summary)
- **search_semantic** - Natural language search: "function that handles auth"
- **get_impact_graph** - See all files depending on a symbol before making changes
- **get_file_context** - Get all symbols in a specific file
- **cache_summary** - Cache your own summaries for future queries
- **search_docs** - Search cached documentation
- **retrieve** - RAG-style retrieval: hybrid search + reranking + context assembly

### Mode Selection

| Mode | Use Case | Token Savings |
|------|----------|---------------|
| `auto` (default) | Most searches - full source for top 3, skeleton for rest | ~80% |
| `skeleton` | Exploration, understanding structure | ~90% |
| `summary` | High-level overviews (requires cache_summary first) | ~94% |
| `full` | Only when you need complete implementation details | 0% |

### Best Practices

1. **Use session_id** - Prevents re-sending files you've already seen
2. **Set token_budget** - Default 4000, adjust based on need
3. **Use get_project_map first** - ~200 tokens for full project overview
4. **Cache summaries** - Call cache_summary after understanding key files
5. **Use search_docs** - For library/framework documentation questions
6. **Use retrieve** - For RAG-style context assembly (code + docs in one call)
```

## Cursor

### MCP Config
File: `.cursor/mcp.json`
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

### Cursor Rules
Add to `.cursor/rules/ast-context-cache.mdc`:
```markdown
---
description: Use ast-context-cache for efficient code search
---

When searching code, prefer using the ast-context-cache MCP tools:
- Use get_context_capsule with mode='auto' for most searches
- Use search_semantic for natural language queries
- Use get_impact_graph before making changes to understand blast radius
- Use get_file_context to get all symbols in a specific file
- Use retrieve for RAG-style context assembly (code + docs)
- Always pass session_id for deduplication
- Use search_docs for library/framework documentation
```

## Claude Desktop

### MCP Config
File: `~/Library/Application Support/Claude/claude_desktop_config.json`
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

## Claude Code

### Project Instructions
File: `CLAUDE.md`
```markdown
# Code Context Instructions

Use ast-context-cache MCP server (http://localhost:7821/mcp) for efficient code search.
- get_context_capsule: Search code with token-efficient modes
- get_file_context: Get all symbols in a specific file
- cache_summary: Cache your own summaries
- analyze_dead_code: Find unused code
- search_docs: Search cached documentation
- retrieve: RAG-style retrieval (code + docs → formatted context)
```

## VS Code (GitHub Copilot)

### MCP Config
File: `.vscode/mcp.json`
```json
{
  "servers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```

## JetBrains (AI Assistant)

### MCP Config
File: `.idea/mcp.json`
```json
{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}
```
