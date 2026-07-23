# ast-context-cache Usage Guide

## What It Does

ast-context-cache is an MCP server that provides token-efficient code search for AI coding agents. Instead of reading entire files, it returns only the symbols (functions, classes, types) you need — saving 80-94% of tokens. It also offers **virtual context** (`ctx_*`) for host compaction survival and **structured memory** (`mem_*`) for compact prefs/rules.

**Operator dashboard:** React SPA at `http://localhost:7830/dashboard/` (not Preact/HTMX). Prometheus metrics: `GET http://localhost:7830/metrics`.

Tool list must match `internal/mcp/tools.go` / `tools/list`. See [README — MCP Tools](../README.md#mcp-tools) and [AGENTS.md](../AGENTS.md) for full tables.

## Available Tools

### Core

| Tool | Purpose |
|------|---------|
| `get_context_capsule` | Search code symbols with hybrid BM25+vector search |
| `search_semantic` | Natural language search ("function that handles auth") |
| `get_file_context` | All symbols in a specific file with mode-aware output |
| `get_project_map` | Hierarchical project overview (~200 tokens) |
| `get_impact_graph` | Find all files depending on a symbol |
| `index_status` | Check if a project is indexed |
| `search_docs` | Search cached library/framework documentation |
| `list_doc_sources` | List tracked documentation sources (read-only) |
| `retrieve` | RAG-style retrieval: hybrid search + reranking + context assembly; optional `include_memory` |
| `fetch_context` | Recover virtual context by `ctx_*` ref(s) |
| `list_context` | List virtual context refs (metadata) |
| `search_context` | Find virtual context when refs are lost |
| `recall_memory` | Compact structured facts/procedures (`mem_*`) |

### Extended

| Tool | Purpose |
|------|---------|
| `index_files` | Index source files for searching |
| `cache_summary` | Cache your summaries for future queries |
| `store_context` | Offload bulky notes → `ctx_*` (optional `kind=kv_repair`, `extract_memory`) |
| `flush_context` | Delete virtual context; free quota |
| `store_memory` | Compact temporal fact or procedural rule → `mem_*` |
| `forget_memory` | Invalidate structured memory |
| `report_kv_repair_event` | Report KV cache miss/quality signal |
| `analyze_dead_code` | Find unused functions/classes/imports |
| `analyze_complexity` | Calculate cyclomatic complexity |
| `export_bundle` | Export indexed code as portable bundle |
| `import_bundle` | Import a previously exported bundle |
| `fetch_doc` | Fetch, register, and return cached doc content |
| `add_doc_source` | Track a doc URL for async background caching |
| `remove_doc_source` | Remove a tracked documentation source |
| `update_doc_source` | Manually refresh a documentation source |

### Complete

| Tool | Purpose |
|------|---------|
| `execute_code` | Run JS code against search results (sandboxed) |

## Virtual context vs structured memory

| Need | Tools | Refs |
|------|-------|------|
| Long plans, diffs, analysis | `store_context` / `fetch_context` / `list_context` / `search_context` / `flush_context` | `ctx_*` |
| Prefs, conventions, short rules | `store_memory` / `recall_memory` / `forget_memory` | `mem_*` |

Use the same **`session_id`** for both. Prefer **`recall_memory`** over **`fetch_context`** for preferences.

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

# RAG-style retrieval (code + docs → formatted context)
retrieve(query="how does authentication work", project_path="/path/to/project")
retrieve(query="error handling patterns", project_path="/path", format="xml", include_memory=true)

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

### 5. Memory and compaction
```
store_memory(kind="fact", session_id="ses_abc123", subject="user.shell", object="fish")
store_context(content="long plan...", session_id="ses_abc123", label="auth plan")
# after compaction:
recall_memory(session_id="ses_abc123", query="shell")
fetch_context(refs=["ctx_..."], session_id="ses_abc123")
```

## Mode Selection

| Mode | Token Usage | Best For |
|------|-------------|----------|
| `auto` | ~20% | Most searches (default) |
| `skeleton` | ~10% | Exploring unfamiliar code |
| `summary` | ~6% | High-level overviews |
| `full` | 100% | Complete implementation details |

## Token savings tracking

Measured on **`get_context_capsule`**, **`get_file_context`**, **`search_semantic`**, **`retrieve`**, and **`execute_code`**:

`tokens_saved = max(0, full_source_baseline − tokens_returned) + session_dedup` (search tools); `execute_code` uses `data_baseline_tokens − tokens_used`.

Each response includes **`tokens_saved`**, **`tokens_used`**, and baseline fields. Doc tools (`fetch_doc`, `search_docs`) and indexing calls do not increment dashboard totals. Use **`mode=auto`** or **`skeleton`**; **`mode=full`** saves ~nothing. Pass **`session_id`** on all context tools for dedup credit.

**Code-mode scripts:** Search tools may return **`code_script_hints`**. See [README](../README.md#code-mode-scripts) and [scripts/code-mode/README.md](../scripts/code-mode/README.md).

## Supported Languages

- Python (.py)
- JavaScript/TypeScript (.js/.jsx/.ts/.tsx)
- Go (.go)
- Bash (.sh)
- Fish (.fish)
- YAML (.yml/.yaml)
