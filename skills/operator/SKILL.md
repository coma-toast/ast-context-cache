# ast-context-cache Operator Skill

## When to Use

When configuring or operating the ast-mcp server (not day-to-day MCP search). Use when the user asks about:
- Embedding backends (ONNX, Ollama, HTTP, OpenAI/LiteLLM)
- Dashboard UI (port 7830)
- Virtual context limits, flush, or metrics
- Log file indexing or retention
- Watcher ignore globs, pause/start/delete watchers
- Health checks or server restart after config changes

For MCP search workflows, point agents to [usage/SKILL.md](../usage/SKILL.md).

## Embedding backends

Vectors must stay **768-dimensional** L2-normalized to match the index. Changing model or dimension requires re-index or clearing vectors.

| `EMBED_BACKEND` | Use case | Main env vars |
|-----------------|----------|---------------|
| `onnx` (default) | Local; `make setup` downloads model + tokenizer | `MODEL_DIR` |
| `ollama` | Ollama with 768-d model (e.g. `nomic-embed-text`) | `OLLAMA_HOST`, `OLLAMA_EMBED_MODEL` |
| `http` | Custom `POST` JSON embed service | `EMBED_HTTP_URL`, `EMBED_HTTP_BEARER` |
| `openai` / `litellm` | OpenAI-compatible `POST /v1/embeddings` | `EMBED_OPENAI_BASE_URL`, `EMBED_OPENAI_MODEL`, `EMBED_OPENAI_API_KEY`, `EMBED_OPENAI_DIMENSIONS` |
| `docker` | [Docker Model Runner](https://docs.docker.com/ai/model-runner/) on port **12434** | `EMBED_DOCKER_URL`, `EMBED_DOCKER_MODEL`, `EMBED_DOCKER_DIMENSIONS` |

See [`docker/README.md`](../../docker/README.md): enable Model Runner + TCP 12434, `docker model pull ai/qwen3-embedding`, `EMBED_BACKEND=docker`, then re-index.

- **Process env** must export these vars for whatever starts `ast-mcp`.
- **Dashboard → Settings:** same keys saved to `~/.astcache/usage.db`; **non-empty env always overrides** SQLite.
- **Restart ast-mcp** after changing embedding settings.
- Confirm: `GET http://localhost:7821/health` and `GET /embed/health`.

See [README — Embedding backends](../../README.md#embedding-backends).

## Dashboard (http://localhost:7830)

Open after `make run` or `ast-mcp dash`. Panels update automatically when queries run or indexing changes.

### Header

| Element | Meaning |
|---------|---------|
| Health bar (center) | Embedder state, **queue** mini-gauge, throughput, cache hit %, heap, uptime |
| Project dropdown | Filters stats, charts, index health, and recent **MCP** activity to one repo |
| Settings (gear) | Embedding backend, ignore globs, log indexing, log retention, pin/unpin, doc sources |

### Sections (top to bottom)

| Section | What it shows |
|---------|----------------|
| **Query activity** | MCP query counts, **tokens saved** (code-context tools only; sublabel: 30d total, avg/day, dedup, vs files), **Virtual context** (active inventory, 30d stored vs accessed, utilization, orphans), avg duration, sessions |
| **Index & runtime** | Corpus scale bars; **embed queue** ring gauge + priority/background bars + worker dots; vectors; watchers (pause/start/delete); disk/memory (server-wide) |
| **Activity** | Time series (daily/hourly, queries vs tokens saved) |
| **Symbol / language / tool / imports** | **Tool performance** table + charts (calls, **CPU**, avg latency, tokens saved); Top imports |
| **Recent activity** | Collapsible **MCP tool calls** vs **Indexing activity** (fsnotify reindex/delete) |

### Settings (operators)

- **Pin project** — priority embedding queue, watchers stay warm longer, slower vector unload when idle
- **Virtual context** — max notes/tokens per session and globally; limit policy (`reject` / `lru_session`); **Flush all virtual context** button; env `AST_CONTEXT_*` overrides non-empty values on restart
- **Watcher ignore globs** — JSON array; applied after `IsCodeFile`
- **Index .log / .txt** — FTS/BM25 only, no embeddings
- **Log retention** — optional `.log` cleanup under absolute roots (dry-run first)
- **Embedding backend** — persists env-equivalent keys (env wins on restart)

### Helping users interpret gauges

- **Tokens saved (today)** — sums `tokens_saved` from **`get_context_capsule`**, **`get_file_context`**, **`search_semantic`**, and **`retrieve`** only. **`fetch_doc` / `search_docs` / `index_*`** do not contribute; **0 today** with heavy doc activity is expected. Sublabel shows 30d total, **avg/day**, dedup, and vs-files. **`mode=full`** calls save ~nothing.
- **Virtual context card** — separate from Tokens saved. **Active inventory** = notes still on disk; **30d stored** = `store_context` volume; **30d accessed** = `fetch_context` + `search_context`; **utilization** = accessed/stored; **orphans** = never fetched. **`GET /api/context-stats`** returns JSON for the same rollup (includes nested **`kv_repair`**). **`POST /api/flush-context`** with `{"all":true}` or `{"session_id":"..."}` (also in Settings).
- **KV repair card** (Memory tab) — golden text archives for quantized KV recovery. **Archives active** = notes with `kind=kv_repair` or tag `kv_repair`; **30d repairs** = `fetch_context` with `repair_reason`; sublabel splits **miss / quality / manual** from `report_kv_repair_event` + repair fetches; **utilization** = repairs / archives stored; **orphans** = archives never fetched for repair. **`GET /api/kv-repair-stats`** (`?project_id=` optional) returns the rollup JSON.
- **Embed queue ring** — fill vs combined capacity (priority 128 + background 2048). Green → orange → red as backlog grows.
- **Pinned projects** — use when one repo should index faster under load.

MCP coding agents normally use MCP tools only; mention the dashboard when the user asks about indexing progress, embeddings, or server health.

## Shell helper

After `make install`:

```bash
ast-mcp start|stop|restart|status|health|log|build|dash
```

## Rebuild after source changes

```bash
make build && ast-mcp restart
```

Cursor: skill `ast-context-cache-rebuild` in `.cursor/skills/ast-rebuild/`.

## SQLite WAL runbook

The code index lives in **`index.db`** (WAL mode). **`index.db-wal`** is the file that grows under embed load — not `usage.db-wal` (typically small). Large index WAL can block TRUNCATE checkpoints and make `ast-mcp stop` appear hung.

**Symptoms:** Dashboard shows **Compacting database WAL** for minutes; `index.db-wal` is hundreds of MB or GB; `GET /api/wal-status` shows `last_busy=1`; logs repeat TRUNCATE with `busy=1`.

**Emergency fix (server stopped):**

```bash
ast-mcp stop   # waits 5s then SIGKILL if checkpoint stuck
sqlite3 ~/.astcache/index.db "PRAGMA wal_checkpoint(TRUNCATE);"
ast-mcp start
```

If `ast-mcp stop` hangs, kill the listener: `kill -9 $(lsof -t -iTCP:7821 -sTCP:LISTEN)` then run the offline checkpoint above.

**Prevention (built-in):** Per-DB WAL metrics on dashboard/API (`index_wal_bytes`, `usage_wal_bytes`, `context_wal_bytes`). PASSIVE on **index** every 30s when index WAL &gt; 32 MB; TRUNCATE when index WAL &gt; 64 MB (quiesces index pool — closes readers, fresh conn); defer TRUNCATE until embed queue idle (up to 2 min); force RESTART+TRUNCATE at 128 MB or after 3 busy streaks; **5m/15m backoff** when TRUNCATE stays busy (no 90s retry storm). Worker throttle uses **index** WAL: 128 MB → 4 workers, 256 MB → 2.

**Dashboard progress:** Amber **Compacting database WAL** banner with phase, elapsed time, index WAL shrink, progress bar. After 2 min with `busy=1`, shows **TRUNCATE blocked — deferring until readers idle**. **Checkpoint WAL now** or `POST /api/wal-checkpoint`; status via `GET /api/wal-status` (includes per-DB WAL bytes).

**Logs:** When started from a TTY, ast-mcp logs to `~/.astcache/ast-mcp.log` by default. **mcp-local** may log to `~/.mcp-local/ast-context-cache.log`; dashboard Logs tab picks the newest file. Override with `AST_MCP_LOG_PATH`.

## WTG / multi-worktree projects

When using [wtg](https://github.com/coma-toast/wtg) with `~/spaces/<workspace>/<repo>` checkouts, ast-mcp:

- **Discovers** git repos under `spaces.root_dir` from `~/.config/wtg/config.yaml` (default `~/spaces`) plus indexed paths
- **Labels** projects as `slapi · nightly` (repo · workspace) in the dashboard dropdown and Settings list
- **Starts watchers** for WTG space checkouts on startup (workspace label present)
- **Excludes** repos matching `project_exclude_paths` (Settings) or `discovery.exclude` in WTG config — hidden from the project filter and skipped during auto-discovery

Each worktree path is a separate `project_path` for MCP — pass the absolute checkout root (e.g. `~/spaces/nightly/slapi`). Override config with `WTG_CONFIG`.

Full detail: [README](../../README.md).
