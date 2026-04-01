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

## Quick Reference

```
1. Check if indexed: index_status(project_path="/path/to/project")
2. Index if needed: index_files(path="/path/to/project", project_path="/path/to/project")
3. Search: get_context_capsule(query="function_name", project_path="/path/to/project", mode="auto")
4. Before changes: get_impact_graph(symbol="ClassName", project_path="/path/to/project")
5. RAG retrieve: retrieve(query="how does auth work", project_path="/path/to/project")
6. Search docs: search_docs(query="React hooks")
```

## Tool Selection Guide

| Need | Tool | Mode |
|------|------|------|
| Find specific function/class | `get_context_capsule` | `auto` |
| Explore unfamiliar code | `get_context_capsule` | `skeleton` |
| Search by meaning | `search_semantic` | - |
| Understand project structure | `get_project_map` | depth=2 |
| Get all symbols in a file | `get_file_context` | `auto` |
| Check change impact | `get_impact_graph` | - |
| Find dead code | `analyze_dead_code` | - |
| Find complex code | `analyze_complexity` | - |
| Cache your findings | `cache_summary` | - |
| **RAG retrieval** | `retrieve` | - |
| Search documentation | `search_docs` | - |
| Add doc source | `add_doc_source` | - |
| List doc sources | `list_doc_sources` | - |
| Update doc source | `update_doc_source` | - |
| Remove doc source | `remove_doc_source` | - |

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

# Include full source code
retrieve(query="middleware implementation", project_path="/path/to/project", include_source=true)
```

## Documentation Tools

Track and search external documentation (like Context7):

```
# Add a documentation source
add_doc_source(name="React", type="markdown", url="https://raw.githubusercontent.com/facebook/react/main/CHANGELOG.md", version="18")

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

## Token Optimization Tips

1. Always use `mode="auto"` for most searches (~80% savings)
2. Use `session_id` to avoid re-sending seen files
3. Set `token_budget` to control response size
4. Cache summaries after understanding code
5. Use `get_project_map` first for new projects (~200 tokens)
