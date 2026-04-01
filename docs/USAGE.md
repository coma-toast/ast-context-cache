# ast-context-cache Usage Guide

## What It Does

ast-context-cache is an MCP server that provides token-efficient code search for AI coding agents. Instead of reading entire files, it returns only the symbols (functions, classes, types) you need — saving 80-94% of tokens.

## Available Tools

| Tool | Purpose | Tier |
|------|---------|------|
| `get_context_capsule` | Search code symbols with hybrid BM25+vector search | Core |
| `search_semantic` | Natural language search ("function that handles auth") | Core |
| `get_project_map` | Hierarchical project overview (~200 tokens) | Core |
| `get_impact_graph` | Find all files depending on a symbol | Core |
| `index_status` | Check if a project is indexed | Core |
| `index_files` | Index source files for searching | Extended |
| `cache_summary` | Cache your summaries for future queries | Extended |
| `analyze_dead_code` | Find unused functions/classes/imports | Extended |
| `analyze_complexity` | Calculate cyclomatic complexity | Extended |
| `export_bundle` | Export indexed code as portable bundle | Extended |
| `import_bundle` | Import a previously exported bundle | Extended |
| `execute_code` | Run JS code against search results (sandboxed) | Complete |

## Recommended Workflow

### 1. First Time With a Project
```
1. get_project_map(project_path="/path/to/project", depth=2)
   → Understand directory structure (~200 tokens)

2. index_files(path="/path/to/project", project_path="/path/to/project")
   → Index all source files

3. index_status(project_path="/path/to/project")
   → Verify indexing completed
```

### 2. Finding Code
```
# Specific function/class
get_context_capsule(query="handleAuth", project_path="/path/to/project", mode="auto")

# Natural language search
search_semantic(query="function that validates user input", project_path="/path/to/project")

# Before making changes
get_impact_graph(symbol="UserService", project_path="/path/to/project")
```

### 3. Token Optimization
```
# Exploration mode (90% savings)
get_context_capsule(query="auth", project_path="/path", mode="skeleton")

# Default mode (80% savings)
get_context_capsule(query="auth", project_path="/path", mode="auto")

# With session dedup (prevents re-sending seen files)
get_context_capsule(query="auth", project_path="/path", mode="auto", session_id="ses_abc123")

# With token budget control
get_context_capsule(query="auth", project_path="/path", mode="auto", token_budget=2000)
```

### 4. After Understanding Code
```
# Cache your summary for future queries
cache_summary(
  file="/path/to/file.ts",
  symbol="handleAuth",
  summary="Handles JWT auth with refresh tokens",
  project_path="/path/to/project"
)
```

## Mode Selection

| Mode | Token Usage | Best For |
|------|-------------|----------|
| `auto` | ~20% | Most searches (default) |
| `skeleton` | ~10% | Exploring unfamiliar code |
| `summary` | ~6% | High-level overviews |
| `full` | 100% | Complete implementation details |

## Supported Languages

- Python (.py)
- JavaScript/TypeScript (.js/.jsx/.ts/.tsx)
- Go (.go)
- Bash (.sh)
- Fish (.fish)
