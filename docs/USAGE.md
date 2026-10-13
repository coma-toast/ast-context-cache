# ast-context-cache Usage Guide

## What It Does

ast-context-cache is an MCP server that provides token-efficient code search for AI coding agents. Instead of reading entire files, it returns only the symbols (functions, classes, types) you need — saving 80-94% of tokens. It also offers **virtual context** (`ctx_*`) for host compaction survival, **structured memory** (`mem_*`) for compact prefs/rules, and **subagent handoff** (`hof_*`) so a subagent starts from what its parent already explored.

**Operator dashboard:** React SPA at `http://localhost:7830/dashboard/` (not Preact/HTMX). Prometheus metrics: `GET http://localhost:7830/metrics`.

Tool list must match `internal/mcp/tools.go` / `tools/list`. See [AGENTS.md — All Tools](../AGENTS.md#all-tools) for full tables and [README — Tool tiers](../README.md#tool-tiers-and-per-tool-overrides) for tiers and overrides.

## Available Tools

### Core

| Tool | Purpose |
|------|---------|
| `get_context_capsule` | Search code symbols with hybrid BM25+vector search |
| `search_semantic` | Natural language search ("function that handles auth") |
| `get_file_context` | All symbols in a specific file with mode-aware output |
| `get_project_map` | Hierarchical project overview (~200 tokens) |
| `get_impact_graph` | Find all files depending on a symbol |
| `diff_impact` | Blast radius of a branch or a GitHub PR (`pr`) |
| `check_symbol_exists` | Whether a name is declared anywhere, and where |
| `check_deletion_safety` | Removed symbols still referenced elsewhere vs safe to delete |
| `index_status` | Check if a project is indexed |
| `search_docs` | Search cached library/framework documentation |
| `list_doc_sources` | List tracked documentation sources (read-only) |
| `retrieve` | RAG-style retrieval: hybrid search + reranking + context assembly; optional `include_memory` |
| `fetch_context` | Recover virtual context by `ctx_*` ref(s) |
| `list_context` | List virtual context refs (metadata) |
| `search_context` | Find virtual context when refs are lost |
| `recall_memory` | Compact structured facts/procedures (`mem_*`) |
| `handoff` | Delegate to a subagent: `create`, `complete`, `collect`, `list`, `status`, `flush` |
| `open_handoff` | Child entry point: `open`, `expand`, `resume` |
| `scratchpad` | Shared findings and advisory claims across a handoff tree: `post`, `read`, `retract`, `claim`, `release` |

### Extended

| Tool | Purpose |
|------|---------|
| `index_files` | Index source files for searching |
| `cache_summary` | Cache your summaries for future queries |
| `store_context` | Offload bulky notes → `ctx_*` (optional `kind=kv_repair`, `extract_memory`) |
| `edit_context` | Edit a note in place, keeping its `ctx_*` ref (`append`/`replace`/`delete`/`rewrite`/`revert`); reports `tokens_reclaimed`. Prefer it over flush + re-store, and avoid compaction to make room — [`docs/context-edit.md`](context-edit.md) |
| `define_context_fn` / `apply_context_fn` | Define and re-invoke a named reusable transform over notes — [`docs/context-fn.md`](context-fn.md), off by default |
| `flush_context` | Delete virtual context; free quota |
| `store_memory` | Compact temporal fact or procedural rule → `mem_*` |
| `forget_memory` | Invalidate structured memory |
| `report_kv_repair_event` | Report KV cache miss/quality signal |
| `analyze_dead_code` | Find unused functions/classes/imports |
| `analyze_complexity` | Calculate cyclomatic complexity |
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

## Host context clearing

Hosts clear old tool results from the chat window on their own (Claude Code context editing, compaction, Cursor summarization). ast-context-cache plans for that rather than fighting it:

- **Self-offloading results.** With `feature_result_offload` on (`AST_FEATURE_RESULT_OFFLOAD`), a `get_context_capsule`, `get_file_context`, `search_semantic` or `retrieve` result over `result_offload_threshold` tokens (default 2000, `AST_RESULT_OFFLOAD_THRESHOLD`) is stored whole as a `ctx_*` note of kind `offload`. The response keeps the top results and starts with a stub such as `[ctx_1a2b3c4d5e6f get_context_capsule auth flow, 9.4k tok — head shown, fetch_context for all]`. With `output=json` the stub is the body's first field, `offloaded`, so the head stays valid JSON. Pass `offload=false` on the call to get the full result inline. Stub grammars: [stubs.md](stubs.md).
- **The stub is what survives.** When the host clears the result, the stub line is enough to get it back: `fetch_context(refs=["ctx_…"])` returns the full text.
- **Separate budget.** Offload notes do not count toward the session or global virtual-context quotas. They share their own cap, `context_offload_max_tokens_global` (default 500,000, `AST_CONTEXT_OFFLOAD_MAX_TOKENS_GLOBAL`); when it fills, the least recently used offloads are evicted first.
- **24-hour TTL.** An offload note not fetched for `offload_ttl_hours` (default 24, `AST_OFFLOAD_TTL_HOURS`) is deleted by an hourly job. `fetch_context` on that ref then returns an `expired` entry with `status: "expired"` and the `tool` and `args` that produced it, so you can re-run the call instead of guessing. Expired refs are remembered for 30 days.
- **Keep what matters.** An offload is a cache, not a note. If a result is worth keeping past the TTL, copy what you need into a `store_context` note (or a `store_memory` fact) of your own.

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

### 6. Delegating to a subagent
```
# parent: snapshot what you explored, then paste the stub into the subagent prompt
handoff(action="create", session_id="ses_abc123", project_path="/path", brief="Find why X fails; don't edit.",
        label="X failure", pointers=[{"key": "internal/x/x.go", "note": "start here"}])
→ stub: [handoff hof_…] X failure — call open_handoff first

# child: open first, use the returned session_id everywhere, finish with complete
open_handoff(handoff="hof_…", project_path="/path")
handoff(action="complete", session_id="<child session_id>", content="<full result>", status="done", summary="…")
→ stub: [result ctx_… for hof_…] done — <summary>   (output as the final message)

# parent: fan in, or recover refs after compaction
handoff(action="collect", session_id="ses_abc123")
handoff(action="list", session_id="ses_abc123")
```

Use `mode="fork"` only when the host spawned a fork that inherited the parent's window. Parallel children share one stub and coordinate with `scratchpad` (claims are advisory). Full guide: [handoff.md](handoff.md).

## Mode Selection

| Mode | Returns | Best For |
|------|---------|----------|
| `auto` | Full source for the top 3 hits, skeletons for the rest | Most searches (capsule default) |
| `skeleton` | Signatures and types only | Exploring unfamiliar code (`search_semantic` / `get_file_context` default) |
| `summary` | Your cached summaries | High-level overviews; read source before editing |
| `full` | Source for every hit | Complete implementation details |
| `edit` | The top hit's exact source plus skeletons of the functions it calls (`get_file_context`: the `symbol` you name) | Right before editing one symbol |

With `feature_mode_v2` off, `auto` falls back to the 4.x score-ratio rule, and `edit` returns full source for the top hit (or `symbol`) and skeletons for the rest.

## Relevance floor

With `feature_relevance_floor` on, `get_context_capsule`, `search_semantic` and `retrieve` drop hits that are not worth their tokens:

- **Relative cut.** Hits scoring below `min_relative_score` of the top hit's score are withheld and counted in `withheld` (`{count, tokens}`; the summary line shows `withheld N (X tok)`).
- **No match.** When even the top hit is weak, the response has no results and `no_match` (`{best_score, hint}`). A hybrid top hit is strong when both BM25 and the vector search found it, its vector similarity reaches the vector floor, or it covers enough of the query's terms. A `search_semantic` top hit needs the vector floor.
- **Collapse.** Test, mock, vendored, generated and same-signature duplicate hits fold into `collapsed` under the result they repeat, unless the query is about tests or the call passes `collapse=false`.

| Setting | Env override | Default |
|---------|--------------|---------|
| `relevance_min_relative` | `AST_RELEVANCE_MIN_RELATIVE` | `0.7` (per call: `min_relative_score`) |
| `relevance_vector_min` | `AST_RELEVANCE_VECTOR_MIN` | `0.45` with the local ONNX model (`EMBED_BACKEND=onnx`); `0.30` with any other embedding backend |
| `relevance_coverage_min` | `AST_RELEVANCE_COVERAGE_MIN` | `0.5` |

The vector floor depends on the embedder because cosine scales differ between models: `0.45` is the token benchmark sweep's value for all-mpnet-base-v2, and `0.30` is a looser floor for backends that have not been swept. An env value wins over the setting, and the setting over the default.

To tune for your embedder, run the benchmark's threshold sweep. It reruns the fixture's code-search scenarios for each setting in a grid and prints the expected hits found, scenarios that lost recall against the baseline (`lost`), negative queries flagged as no-match, and total tokens. Pick the row with `lost 0` and every negative flagged, then the fewest tokens:

```bash
make bench-tokens                                              # hash embedder (tests run it with relevance_vector_min=0.22)
AST_TOKENBENCH_EMBEDDER=onnx make bench-tokens                 # the real ONNX model from MODEL_DIR or ./model
AST_TOKENBENCH_EMBEDDER=onnx AST_RELEVANCE_SWEEP=1 make bench-tokens   # threshold sweep with the ONNX model
```

## Token savings tracking

Measured on **`get_context_capsule`**, **`get_file_context`**, **`search_semantic`**, **`retrieve`**, and **`execute_code`**:

`tokens_saved = max(0, full_source_baseline − tokens_returned) + session_dedup` (search tools); `execute_code` uses `data_baseline_tokens − tokens_used`.

For the four code tools, the query log and dashboard count `tokens_returned` on the text actually sent (after rendering as text, locations or JSON, and after offload), not on the JSON body. The dashboard splits the total into ledgers:

| Ledger | Meaning |
|--------|---------|
| Compression | Full source of every returned symbol minus the tokens actually returned |
| Dedup | Tokens not re-sent because the session already received those symbols |
| Conservative | Each returned symbol plus 20 lines above and below, merged per file, minus the tokens returned: a floor for the compression figure |
| Virtual | Tokens written to virtual context and memory, and later fetched or recalled; not counted as savings |

**Tokens saved** on the Overview is compression + dedup. Rows logged before 5.0 counted tokens as bytes/4 and are shown as estimated.

The `output=json` body includes **`tokens_saved`**, **`tokens_used`**, and baseline fields. Doc tools (`fetch_doc`, `search_docs`) and indexing calls do not increment dashboard totals. Use **`mode=auto`** or **`skeleton`**; **`mode=full`** saves ~nothing. Pass **`session_id`** on all context tools for dedup credit.

**Code-mode scripts:** Search tools may return **`code_script_hints`**. See [AGENTS.md](../AGENTS.md#code-mode-scripts) and [scripts/code-mode/README.md](../scripts/code-mode/README.md).

## Supported Languages

- Python (.py)
- JavaScript/TypeScript (.js/.jsx/.ts/.tsx)
- Go (.go)
- Bash (.sh)
- Fish (.fish)
- YAML (.yml/.yaml)
