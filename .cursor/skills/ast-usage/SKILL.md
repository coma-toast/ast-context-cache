---
name: ast-context-cache-usage
description: Use when searching, exploring, or analyzing code with ast-context-cache MCP tools (get_context_capsule, retrieve, search_semantic, modes, filters, bundles).
---

# ast-context-cache Usage

## When to Use

When you need to search, understand, or analyze code in a project efficiently.
Use this skill when the user asks to:
- Find functions, classes, or symbols
- Understand code structure
- Find dependencies or impact of changes
- Search code by meaning/intent
- Find unused or complex code
- Search documentation for libraries/frameworks
- Share indexed code with another machine (bundles)
- Run code analysis against search results

## Quick Reference

```
1. Check if indexed: index_status(project_path="/path/to/project")
2. Index if needed:  index_files(path="/path/to/project", project_path="/path/to/project")
3. Orient:           get_project_map(project_path="/path/to/project", depth=2)
4. Search:           get_context_capsule(query="function_name", project_path="...", mode="auto")
5. Before changes:   get_impact_graph(symbol="ClassName", project_path="/path/to/project")
6. RAG retrieve:     retrieve(query="how does auth work", project_path="/path/to/project")
7. Search docs:      search_docs(query="React hooks")
```

## Tool availability (tiers)

If `index_files`, `execute_code`, or other tools are missing from `tools/list`, the server tier is too low or the tool is disabled in `~/.astcache/tools.json`. Agents cannot change tier—ask the user to set `AST_MCP_TIER` / edit overrides and restart ast-mcp. See [README](../../README.md#tool-tiers-and-per-tool-overrides).

## Tool Selection Guide

| Need | Tool | Notes |
|------|------|-------|
| Find specific function/class | `get_context_capsule` | mode=`auto` |
| Explore unfamiliar code | `get_context_capsule` | mode=`skeleton` |
| Search by meaning/intent | `search_semantic` | Natural language; optional `doc_type` |
| Understand project structure | `get_project_map` | depth=2 (~200 tokens) |
| Get all symbols in a file | `get_file_context` | mode=`auto` |
| Check change impact | `get_impact_graph` | Before modifying symbols |
| Find dead code | `analyze_dead_code` | kind filter optional |
| Find complex code | `analyze_complexity` | threshold default=10 |
| Cache your findings | `cache_summary` | Enables summary mode |
| RAG retrieval (code+docs) | `retrieve` | Best single-call context |
| Search documentation | `search_docs` | FTS over cached docs |
| Add doc source | `add_doc_source` | markdown, html, json |
| List doc sources | `list_doc_sources` | core tier |
| Refresh a doc source | `update_doc_source` | Pass doc id |
| Remove a doc source | `remove_doc_source` | Pass doc id |
| Share code index | `export_bundle` | Creates .astbundle |
| Load shared index | `import_bundle` | No re-indexing needed |
| Process search results | `execute_code` | complete tier; JS sandbox |

## Modes

| Mode | Token Usage | Best For |
|------|-------------|----------|
| `auto` | ~20% of full | Most searches (recommended default) |
| `skeleton` | ~10% of full | Exploring unfamiliar code structure |
| `summary` | ~6% of full | High-level overviews (requires `cache_summary` first) |
| `full` | 100% | When you need complete implementation details |

## RAG Retrieval

The `retrieve` tool does full RAG-style retrieval in one call:
- Hybrid search (BM25 + vector) across code AND docs
- Reranks and deduplicates results
- Assembles context within token budget
- Returns formatted output (markdown/xml/json)

```
retrieve(query="how does authentication work", project_path="/path/to/project")
retrieve(query="auth handler", project_path="/path/to/project", path_prefix="internal/auth", language="go")
```

## Documentation Tools

```
add_doc_source(name="React", type="markdown", url="https://...", version="18")
search_docs(query="useState hook", limit=5)
list_doc_sources()
update_doc_source(id=1)
remove_doc_source(id=1)
```

## Optional Search Filters

For `get_context_capsule`, `search_semantic`, and `retrieve`:

| Parameter | Purpose |
|-----------|---------|
| `path_prefix` | Only symbols under this path (project-relative) |
| `language` | `go`, `python`, `typescript`, `javascript`, `yaml`, etc. |
| `kinds` | Comma-separated symbol kinds |
| `kind` | Single kind filter |
| `doc_type` | On `search_semantic` only: e.g. `code`, `doc` |

## Supported languages

Python, JavaScript/JSX, TypeScript/TSX, Go, Bash, Fish, YAML.

## Pipeline Observability

- **get_context_capsule** — `pipeline`: `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse`
- **retrieve** — `stats`: hybrid counts, `after_dedup`, timings (`code_retrieve_ms`, etc.)

## Token Optimization Tips

1. Use `mode="auto"` for most searches
2. Pass `session_id` on `get_context_capsule` / `search_semantic`
3. Set `token_budget` (default 4000)
4. `cache_summary` before `mode=summary`
5. `get_project_map` first on new projects
6. Narrow with `path_prefix` + `language` before ranking

Canonical copy: [skills/usage/SKILL.md](../../skills/usage/SKILL.md)
