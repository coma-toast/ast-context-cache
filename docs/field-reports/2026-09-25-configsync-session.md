# Field report: ast-context-cache issues found during a configSync session (2026-09-25)

Status: **every issue has a fix in review** (2026-10-02), as one stack of PRs:

| Issue | Fix |
|---|---|
| #1 search tools take ~40s | #94 |
| #2 class methods missing, #9 `cache_summary` accepts unindexed symbols | #98 |
| #3 deleted files stay indexed, #5 symlinks indexed twice | #99 |
| #4 no directory excludes | #100 |
| #6 `get_impact_graph` local imports / substring matches | #97 |
| #7 `extract_memory`, #8 `forget_memory` | #96 |
| #10 `index_files` blocks, #11 `search_docs` floor, #12 housekeeping | #101 |

The report below is as written on 2026-09-25; no code in this repo was changed while collecting it.

## Environment

| | |
|---|---|
| Server version | `3.0.22` (`VERSION`), HEAD `b06566d` on branch `NO-TICKET-overview-declutter-and-quick-delete` (uncommitted work present — unrelated to this report) |
| Binary | `/Users/jason/git/ast-context-cache/ast-mcp`, supervised by mcp-local, log `~/.mcp-local/ast-context-cache.log` |
| Endpoint | `http://localhost:7821/mcp` (streamable HTTP; `initialize` returned no `Mcp-Session-Id`, calls worked statelessly) |
| Embedder | `embed_mode: openai`, `embed_model: vm/nomic-embed-text-v1.5.Q4_K_M.gguf` via LiteLLM. Measured healthy and fast: 0.08s direct to vm:8086, 0.45s through LiteLLM — **not** the bottleneck below |
| Project | `/Users/jason/configSync` (no git; Syncthing-synced) — grew from 1448 files / 23037 nodes to **2285 files / 41609 nodes** after a catch-up re-index |
| DB | `~/.astcache/index.db` = **4.18 GB**; WAL peaked at **3.93 GB** during catch-up |
| MCP client | Claude Code (tool calls time out client-side at roughly 30–40s) |

Severity key: **High** = tool returns wrong answers or is unusable; **Medium** = degraded/misleading; **Low** = cleanup/polish.

---

## 1. [High] Search tools take ~40s per call and time out in the MCP client

`get_context_capsule`, `search_semantic`, and `retrieve` all timed out from Claude Code — first during a catch-up re-index, and **still after it finished** (embed queue idle for 2+ minutes, CPU down to 17%).

Measured directly against the endpoint (no client timeout) once indexing was idle:

```
index_status                              HTTP 200 in  1.38s
get_context_capsule (query=build_hosts_dict, mode=skeleton, token_budget=800)   HTTP 200 in 38.62s
search_semantic (query="load config.yaml hosts", path_prefix=agent, limit=5)    HTTP 200 in 43.51s
```

The capsule response's `pipeline` block: `bm25_candidates: 1, vector_candidates: 60, hybrid_after_fuse: 60`.

**Repro** (JSON-RPC straight to the server):
```bash
U=http://localhost:7821/mcp
curl -s -o /tmp/out.json -w "%{time_total}s\n" -H 'Content-Type: application/json' \
  -H 'Accept: application/json,text/event-stream' -X POST $U -d \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_context_capsule","arguments":{"query":"build_hosts_dict","project_path":"/Users/jason/configSync","mode":"skeleton","token_budget":800}}}'
```

**Expected:** sub-second to a few seconds for a 41k-node project.
**To investigate:** whether vector search is a brute-force scan over all nodes; whether `path_prefix` is applied after retrieval rather than in the query; lock contention with the post-idle "running maintenance" step (log also shows `WAL TRUNCATE deferred: readers busy`); DB size (see #4). `/health` reported `healthy`/`ready` the whole time, so health doesn't reflect degraded query latency — consider exposing p50/p95 search latency there.

## 2. [High] `check_symbol_exists` returns false negatives for class methods

Python methods aren't indexed as their own symbols — `get_file_context` on
`agent/skills/model_manager/model_manager/clients/llamacpp.py` returns a single `class LlamaCppClient` symbol with methods inlined into its skeleton. Consequences:

| Query | Actual | Tool said |
|---|---|---|
| `to_litellm_params` | method of `ModelData`, `litellm_sync.py:331` | `exists: false` |
| `LlamaCppClient.load_model` (qualified) | exists | `exists: false` |
| `load_model` | methods on `LlamaCppClient`, `OMLXClient`, `LMStudioClient` (+ others) | only 3 unrelated **module-level** functions (llama.cpp tree, webui backend, and a deleted file) |

This is the tool agents are told to use *before trusting a reference* — a false "doesn't exist" is worse than no answer. Same root cause likely limits `get_impact_graph`, `search_*`, and `cache_summary` for methods (see #9).
Also noticed: skeletons truncate multi-line signatures mid-parameter list (`def __init__(self, host:`, `def _list_local_models(self, model_dirs:`).

## 3. [High] Deleted files stay in the index and are re-embedded by catch-up

These files do **not** exist on disk, yet appear in `check_symbol_exists` / `get_impact_graph` / capsule results:

- `skills/sync-litellm-models/litellm_sync.py`, `skills/sync-litellm-models/litellm-verify.py` (parent dir exists but is empty)
- `skills/model-manager/model-manager.py`, `skills/lm-studio-benchmark/run_benchmark.py`
- `llm-benchmark/dashboard.py`, `llm-benchmark/model_registry.py`, `llm-benchmark/switch_model.py`, `llm-benchmark/archive/switch_model.py`
- `restore/terraform/lxc_containers.tf`, `restore/devops/ansible-ui/HostManager.js`

Log shows the catch-up **embedding** deleted files today:
```
182498: 2026/09/24 18:09:13 Embedded 13 symbols from /Users/jason/configSync/llm-benchmark/dashboard.py
188517: 2026/09/25 12:21:30 Embedded 17 symbols from /Users/jason/configSync/restore/terraform/lxc_containers.tf
188523: 2026/09/25 12:21:31 Embedded 13 symbols from /Users/jason/configSync/llm-benchmark/dashboard.py
```
Impact: in a capsule query for `build_hosts_dict`, 3 of the 10 results pointed at deleted files. Catch-up should prune index rows whose path no longer exists instead of re-queuing them.

## 4. [Medium] No way to exclude directories — vendored trees bloat the index

`ShouldSkipDir` (`internal/indexer/indexer.go:38`) only skips dot-dirs and the hardcoded `SkipDirs` map (`indexer.go:30`: node_modules, vendor, venv, env, __pycache__, dist, build, .astcache, target, bower_components, third_party, …). No `.astignore`, `.gitignore`, `.stignore`, env var, or per-project config is honored.

In configSync this indexed: two full llama.cpp source checkouts (`llama-cpp-tq-tom/`, `llama-cpp-tq3/`), `openviking_workspace/…/resources/ast-context-cache/` and `…/temp/…/repository/` (copies of **this repo**), plus `restore/`, `terraform/`, `usb-recovery/` duplicates. Search results for configSync-specific questions are diluted by these, and the catch-up that pulled them in took hours (embed queue ~665 pending batches) and grew `index.db` to 4.18 GB.

Suggested: honor `.gitignore` + `.stignore` + a repo-level `.astignore`, and/or a per-project exclude list settable via MCP/dashboard.

## 5. [Medium] Symlinked files indexed twice

`agent/skills/sync-litellm-models/litellm-sync.py` is a symlink to `litellm_sync.py`. Both are indexed, so every symbol in that file is returned twice (`_effort_lookup`, `ModelData`, `load_config`, …). Resolve real paths and dedupe (or index the symlink as an alias).

## 6. [Medium] `get_impact_graph` misses function-local imports; has substring false positives

- **Misses:** `model_manager` imports `HOSTS` inside method/function bodies (`from model_manager.mm_config import HOSTS`) at ~25 sites in 12 files. `get_impact_graph(HOSTS)` reported none of them; `get_impact_graph(switch_local_model)` reported **0** callers although `clients/llamacpp.py` imports and calls it inside `load_model`. Grep found all of them.
- **False positives:** `get_impact_graph(HOSTS)` and `get_impact_graph(load_config)` both listed `design-system/eslint.config.js` (`target: "eslint/config"`) — looks like substring matching on import targets. `defined_in` for `HOSTS` listed YAML files (`config.yaml`, an Ansible playbook).

## 7. [Medium] `store_context(extract_memory: true)` extracts every line, not just `FACT:`/`RULE:` lines

The tool description says it parses `FACT:/RULE:` lines. A note with 2 `FACT:` + 2 `RULE:` lines produced **22** memory entries — including markdown headings as facts (`## BROKEN NOW`, `## WRONG BEHAVIOR`) and garbled rewrites where `path:line` became `path is line`, e.g.
`bonsai/plugin.py is 33 SERVICE_MODEL=Ternary-Bonsai …` (from `bonsai/plugin.py:33 SERVICE_MODEL=…`).
The 4 intended entries (`mem_10c9e0662947`, `mem_589f9e2bc85b`, `mem_0aa478bc60c0`, `mem_8a5eebb7a357`) were created correctly; the other 18 had to be deleted by hand. Source note: `ctx_fe05bd0e56a6` (session `a5c68d1d-42e7-413b-873c-b1d2f8d5fcd1`).

## 8. [Medium] `forget_memory` silently ignores multiple refs and unscoped refs

| Call | Result |
|---|---|
| `refs: "mem_a,mem_b,…"` (comma-separated, 18 refs) | `invalidated_refs: 0`, no error |
| `refs: ["mem_a","mem_b",…]` (JSON array) | `invalidated_refs: 0`, no error |
| same, plus `scope: session` + `session_id` | `invalidated_refs: 0`, no error |
| single ref, no scope/session | `invalidated_refs: 0`, no error |
| single ref + `scope: session` + `session_id` | `invalidated_refs: 1` ✅ |

Required 18 separate calls. Suggested: accept arrays/comma lists; resolve a `mem_*` ref's scope from the ref itself; return an error (or `not_found` list) for refs that matched nothing instead of a silent 0.

## 9. [Low, suspected] `cache_summary` accepts symbols that aren't indexed

`cache_summary(symbol: "load_model", file: …/clients/llamacpp.py)` and `cache_summary(symbol: "to_litellm_params", file: …/litellm_sync.py)` both returned `status: cached`, but per #2 neither is an indexed symbol. Likely orphaned (never served by `mode: summary`). Not verified — check whether these rows are reachable, and consider rejecting or warning on unknown symbols.

## 10. [Low] `index_files` on a directory blocks until the client times out

`index_files(path=…/agent/skills/model_manager/model_manager)` and `index_files(path=…/agent/skills/sync-litellm-models)` both timed out client-side (while catch-up was running). Indexing did complete server-side (new symbols became visible via `check_symbol_exists`). Suggest returning immediately with a job id/progress (or enqueuing) for directory targets.

## 11. [Low] `search_docs` returns unrelated hits when nothing matches

Query `module __getattr__ PEP 562 lazy attribute` (not in cache) returned 5 results from Tailscale API and ProxmoxVED AGENTS.md docs with scores ~0.016–0.03. No relevance floor or `no_match` signal, so an agent can mistake these for real hits. (`fetch_doc` for the PEP then worked fine.)

## 12. [Low] Housekeeping

- Zero-byte stray DB files in `~/.astcache/`: `ast-cache.db` (Mar 3), `ast.db` (May 29), `astcache.db` (May 29).
- WAL grew to 3.93 GB during catch-up (`embed queue: throttled workers 10 -> 2 (target 10 wal=3.93 GB)`), later truncated to 17 MB. Worth confirming disk-pressure handling on smaller disks.

---

## Worked as expected (for contrast)

`index_status`, `get_project_map`, `get_file_context` (apart from #2's signature truncation), `check_symbol_exists` for module-level functions/classes, `store_context`/`list_context` (apart from #7), `store_memory`/`recall_memory` (project + global scope), `fetch_doc`, and `execute_code` with `script_id: compact-symbol-list` (fed from a capsule result's `code_script_hints`). `diff_impact` and `check_deletion_safety` were not exercised — configSync has no git.
