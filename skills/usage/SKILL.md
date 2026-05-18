# ast-context-cache Usage Skill

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

## Tool Selection Guide

| Need | Tool | Notes |
|------|------|-------|
| Find specific function/class | `get_context_capsule` | mode=`auto` |
| Explore unfamiliar code | `get_context_capsule` | mode=`skeleton` |
| Search by meaning/intent | `search_semantic` | Natural language |
| Understand project structure | `get_project_map` | depth=2 (~200 tokens) |
| Get all symbols in a file | `get_file_context` | mode=`auto` |
| Check change impact | `get_impact_graph` | Before modifying symbols |
| Find dead code | `analyze_dead_code` | kind filter optional |
| Find complex code | `analyze_complexity` | threshold default=10 |
| Cache your findings | `cache_summary` | Enables summary mode |
| RAG retrieval (code+docs) | `retrieve` | Best single-call context |
| Search documentation | `search_docs` | FTS over cached docs |
| Add doc source | `add_doc_source` | markdown, html, json |
| List doc sources | `list_doc_sources` | - |
| Refresh a doc source | `update_doc_source` | Pass doc id |
| Remove a doc source | `remove_doc_source` | Pass doc id |
| Share code index | `export_bundle` | Creates .astbundle |
| Load shared index | `import_bundle` | No re-indexing needed |
| Process search results | `execute_code` | JS sandbox, DATA var |

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
# Basic RAG retrieval
retrieve(query="how does authentication work", project_path="/path/to/project")

# With custom budget and format
retrieve(query="database connection pooling", project_path="/path/to/project", token_budget=2000, format="xml")

# Code only (skip docs)
retrieve(query="error handling patterns", project_path="/path/to/project", include_docs=false)

# Include full source code in results
retrieve(query="middleware implementation", project_path="/path/to/project", include_source=true)

# Narrow by path and language
retrieve(query="auth handler", project_path="/path/to/project", path_prefix="internal/auth", language="go")
```

## Documentation Tools

Track and search external documentation (like Context7):

```
# Add a documentation source
add_doc_source(name="React", type="markdown", url="https://...", version="18")

# Search cached documentation
search_docs(query="useState hook", limit=5)

# List all tracked doc sources
list_doc_sources()

# Manually update a doc source
update_doc_source(id=1)

# Remove a doc source
remove_doc_source(id=1)
```

Doc sources auto-update every hour. Supports `markdown`, `html`, and `json` types.

## Optional Search Filters

For `get_context_capsule`, `search_semantic`, and `retrieve`:

| Parameter | Purpose |
|-----------|---------|
| `path_prefix` | Only symbols under this path (project-relative, e.g. `internal/mcp`) |
| `language` | Filter by language: `go`, `python`, `typescript`, `javascript`, etc. |
| `kinds` | Comma-separated symbol kinds (e.g. `function,method`) |
| `kind` | Single kind filter |

## Code Execution Against Search Results

`execute_code` runs JavaScript in a sandbox. Search results are injected as `DATA`.
Use this to process, filter, or transform results before they enter your context window.

```
# Find all functions with 'handler' in their name from a search
execute_code(
  data=<search results JSON>,
  code="return DATA.filter(r => r.name && r.name.includes('handler')).map(r => r.name)"
)
```

## Bundle Sharing

Export and import indexed code without re-indexing:

```
# Export
export_bundle(project_path="/path/to/project", output_path="/tmp/myproject.astbundle")

# Import on another machine
import_bundle(bundle_path="/tmp/myproject.astbundle")
```

## Pipeline Observability

- **`get_context_capsule`** response includes a `pipeline` object:
  - `bm25_candidates`, `vector_candidates`, `hybrid_after_fuse`
- **`retrieve`** response `stats` includes:
  - Hybrid stage counts: `after_dedup`, `chunks_in_budget`, `tokens_est_all_chunks`
  - Timings: `code_retrieve_ms`, `docs_retrieve_ms`, `dedup_budget_ms`, `search_time_ms`

## Token Optimization Tips

1. Always use `mode="auto"` for most searches (~80% savings)
2. Use `session_id` to avoid re-sending symbols already seen in this session
3. Set `token_budget` to control response size (default 4000)
4. Cache summaries after understanding code — enables cheap `summary` mode later
5. Use `get_project_map` first for new projects (~200 tokens for full overview)
6. Use `path_prefix` + `language` filters to narrow searches before ranking
