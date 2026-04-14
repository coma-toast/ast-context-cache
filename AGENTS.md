# Agent Code Research Guidelines

## MCP Server Tools (Preferred)

When working with codebases that have an MCP server available, **always prefer MCP tools** over direct grep/read/glob:

- **get_context_capsule** - Search code with token-efficient modes (auto, skeleton, summary)
- **search_semantic** - Natural language search: "function that handles auth"
- **get_file_context** - Get all symbols in a specific file (use instead of reading files)
- **get_impact_graph** - See all files depending on a symbol before making changes
- **search_docs** - Search cached library/framework documentation
- **cache_summary** - Cache your own summaries for future queries

### All Tools

#### Core

| Tool | Description |
|------|-------------|
| `get_context_capsule` | BM25+vector hybrid search. Modes: `full`, `skeleton`, `summary`, `auto`. |
| `search_semantic` | Semantic search by meaning using vector embeddings. |
| `get_file_context` | All symbols in a file with mode-aware output. Use instead of reading files. |
| `get_project_map` | Project structure overview (depth 1=dirs, 2=files, 3=symbols). |
| `get_impact_graph` | Blast radius of a symbol -- files that import or depend on it. |
| `index_status` | Check if a project is indexed. Returns file/symbol counts. |
| `search_docs` | Search locally cached documentation by title or content (FTS). |

#### Extended

| Tool | Description |
|------|-------------|
| `index_files` | Index a file or directory. Starts a file watcher for incremental re-indexing. Plain `.log` is not indexed unless enabled in dashboard settings; watcher ignore globs apply to paths that would otherwise be indexed as code. |
| `cache_summary` | Store a summary for a file/symbol for cheap future lookups. |
| `analyze_dead_code` | Find unused functions, classes, and imports. |
| `analyze_complexity` | Calculate cyclomatic complexity to find hard-to-maintain code. |
| `export_bundle` | Export indexed code as a portable `.astbundle` file. |
| `import_bundle` | Import a previously exported bundle without re-indexing. |
| `add_doc_source` | Add a documentation URL to track and cache (markdown, html, json). |
| `remove_doc_source` | Remove a tracked documentation source. |
| `list_doc_sources` | List all tracked documentation sources. |
| `update_doc_source` | Manually refresh a documentation source. |

#### Complete

| Tool | Description |
|------|-------------|
| `execute_code` | Run JavaScript in a sandbox against search results. Only output enters context. |

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
4. **Use get_file_context over read** - Returns structured, mode-aware results
5. **Cache summaries** - Call cache_summary after understanding key files
6. **Use search_docs** - For library/framework documentation questions
7. **Optional filters** - On `get_context_capsule`, `search_semantic`, and `retrieve`, pass `path_prefix`, `language`, and/or `kinds` / `kind` to narrow results to a subtree or language
8. **Pipeline stats** - `get_context_capsule` returns `pipeline` counts; `retrieve` stats include hybrid-stage counts and timings (see README / CLAUDE.md)
9. **Indexing load** - Embeddings go through a bounded **queue** with workers (dashboard: embed queue / active). **Pin** heavy projects in Settings for priority embedding, no idle watcher stop, and warmer vector unload behavior

### Documentation Tools

Track and search external documentation (similar to Context7):

```
add_doc_source(name="React", type="markdown", url="https://...", version="18")
search_docs(query="useState hook", limit=5)
list_doc_sources()
update_doc_source(id=1)
remove_doc_source(id=1)
```

Doc sources auto-update every hour. Supports `markdown`, `html`, and `json` types.

## When MCP Not Available

Use grep/ast_grep only when:
- You know exactly what to search (single keyword)
- Known file location
- MCP server is not running

Avoid:
- Multi-angle searches (use MCP search_semantic instead)
- Cross-module pattern discovery
- Unfamiliar codebase exploration

## Slide / `slidehq` workspaces

When researching or changing code in Slide repositories (for example `common`, `sandbox`, `box`, `cloud`, `agent`), **prefer shared logic and patterns from [`github.com/slidehq/common`](https://github.com/slidehq/common)** over inventing new abstractions in a single repo. Check common for existing utilities, errors, task patterns, retries, and logging before adding parallel implementations elsewhere.
