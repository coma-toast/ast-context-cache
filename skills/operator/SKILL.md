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

## Dashboard (http://localhost:7830/dashboard/)

Open after `make run` or `ast-mcp dash`. The UI is a **Preact + MUI** SPA embedded at `/dashboard/` (root `/` redirects). Updates use **WebSocket `/ws`** (`refresh` panel events + MCP toasts), not SSE or HTMX partials.

**Dev:** `make ui-dev` (Vite on :5173, proxies API/WS) while ast-mcp runs on :7830.

### Header

| Element | Meaning |
|---------|---------|
| Health bar | Embedder state, queue depth, throughput, cache hit %, heap, uptime, version |
| Project filter | Filters stats, index health, memory, recent, and charts to one repo |

### Sidebar tabs

| Tab | What it shows |
|-----|----------------|
| **Overview** | Query activity, virtual context summary, index & runtime (embed queue, corpus, watchers) |
| **Memory** | Virtual context stats, doc sources (add/refresh/delete) |
| **Activity** | Time series (daily/hourly, queries vs tokens saved) |
| **Analytics** | Tool performance table, symbol/language/import charts |
| **Recent** | MCP tool calls vs indexing activity (accessible error expand) |
| **Settings** | Performance, virtual context, embedding backend, watcher, retention, **projects (link/unlink subprojects)**, agent install, MCP tier (read-only) |

### Settings (operators)

- **Link subproject** — container parent + child picker (`POST /api/project-links`); auto-link still runs on index
- **Pin project** — priority embedding queue, watchers stay warm longer
- **Virtual context** — limits + **Flush all**; per-session flush via API `POST /api/flush-context` with `session_id`
- **Watcher ignore globs** — JSON array
- **Embedding backend** — persisted to SQLite; env overrides on restart
- **MCP tier** — read-only card (`AST_MCP_TIER`, `~/.astcache/tools.json`)

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

- **Container linking:** indexing a parent folder (e.g. `~/git`) auto-links already-indexed sub-repos; parent skips duplicate indexing and search includes linked children. Manage links in Settings → Projects (Unlink per child).

Full detail: [README](../../README.md).
