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
- **Virtual context card** — separate from Tokens saved. **Active inventory** = notes still on disk; **30d stored** = `store_context` volume; **30d accessed** = `fetch_context` + `search_context`; **utilization** = accessed/stored; **orphans** = never fetched. **`GET /api/context-stats`** returns JSON for the same rollup. **`POST /api/flush-context`** with `{"all":true}` or `{"session_id":"..."}` (also in Settings).
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

The dashboard reads `~/.astcache/usage.db` (WAL mode). If the WAL file grows very large, HTTP handlers can block while SQLite scans frames — MCP may still respond while the dashboard hangs.

**Symptoms:** Dashboard at http://localhost:7830 stops loading; `usage.db-wal` is hundreds of MB or GB.

**Emergency fix:**

```bash
ast-mcp stop
sqlite3 ~/.astcache/usage.db "PRAGMA wal_checkpoint(TRUNCATE);"
ast-mcp start
```

**Prevention (built-in):** `PRAGMA wal_autocheckpoint=1000`; passive checkpoints every 2m; TRUNCATE when WAL &gt; 256 MB or every 30m; startup TRUNCATE when WAL &gt; 100 MB. **Query retention** (Settings → Query history retention, default 90 days) prunes old `queries` rows daily and checkpoints after deletes. Overview shows WAL size on the Database row.

**Logs:** Default server log is `~/.astcache/ast-mcp.log` (`ast-mcp start`). **mcp-local** logs to `~/.mcp-local/ast-context-cache.log`; the dashboard Logs tab auto-picks the newest log file. Override with `AST_MCP_LOG_PATH`.

## WTG / multi-worktree projects

When using [wtg](https://github.com/coma-toast/wtg) with `~/spaces/<workspace>/<repo>` checkouts, ast-mcp:

- **Discovers** git repos under `spaces.root_dir` from `~/.config/wtg/config.yaml` (default `~/spaces`) plus indexed paths
- **Labels** projects as `slapi · nightly` (repo · workspace) in the dashboard dropdown and Settings list
- **Starts watchers** for WTG space checkouts on startup (workspace label present)
- **Excludes** repos matching `project_exclude_paths` (Settings) or `discovery.exclude` in WTG config — hidden from the project filter and skipped during auto-discovery

Each worktree path is a separate `project_path` for MCP — pass the absolute checkout root (e.g. `~/spaces/nightly/slapi`). Override config with `WTG_CONFIG`.

Full detail: [README](../../README.md).
