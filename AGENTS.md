# Agent Code Research Guidelines

## MCP Server Tools (Preferred)

When working with codebases that have an MCP server available, **always prefer MCP tools** over direct grep/read/glob:

- **get_context_capsule** - Search code with token-efficient modes (auto, skeleton, summary)
- **search_semantic** - Natural language search: "function that handles auth"
- **get_impact_graph** - See all files depending on a symbol before making changes
- **cache_summary** - Cache your own summaries for future queries

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

## When MCP Not Available

Use grep/ast_grep only when:
- You know exactly what to search (single keyword)
- Known file location
- MCP server is not running

Avoid:
- Multi-angle searches (use MCP search_semantic instead)
- Cross-module pattern discovery
- Unfamiliar codebase exploration
