# ast-context-cache

[![Release](https://img.shields.io/github/v/release/coma-toast/ast-context-cache)](https://github.com/coma-toast/ast-context-cache/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

**Give AI coding agents the code they need in a fraction of the tokens.** ast-context-cache is a local MCP server that indexes your repos with tree-sitter and answers agents with symbols, signatures, and summaries instead of whole files. It also keeps an agent's plans alive across chat compaction and hands subagents what their parent already found, so nothing gets read twice. Everything runs on your machine: no cloud, no account, no data leaves it.

![Dashboard overview](docs/images/dashboard-overview.png)

## How it saves tokens

An agent without it reads whole files, greps, re-reads the same code after compaction, and starts every subagent from zero. Each of those is a place ast-context-cache cuts tokens:

| Instead of… | ast-context-cache returns… | Typical saving |
|---|---|---|
| Reading a whole file to find one function | The matching **symbols**, ranked by hybrid BM25 + vector search | Only what matched, not the file around it |
| Reading full source to learn a file's shape | A **skeleton**: signatures and types only (`get_file_context`, `mode=skeleton`) | ~90% fewer tokens |
| Re-reading code you've understood | Your cached **summary** of it (`cache_summary`, then `mode=summary`) | ~94% fewer tokens |
| Full source for every search hit | **`auto` mode**: full source for the top 3 hits, skeletons for the rest | ~80% fewer tokens |
| Sending the same symbol twice in a conversation | Nothing: **session dedup** skips symbols already returned to that `session_id` | 100% of the repeat |
| Grepping callers before an edit | **Impact checks** (`get_impact_graph`, `diff_impact`, `check_deletion_safety`) list exactly what depends on a symbol | One answer instead of many reads |
| Losing a long plan to chat compaction, then rebuilding it | **Virtual context**: `store_context` keeps the plan on disk behind a short `ctx_*` ref; `fetch_context` brings it back | The plan costs a ref until it's needed |
| Rewriting a stored note by flushing and re-storing it | **`edit_context`** edits it in place — same ref, reversible, and it reports `tokens_reclaimed` | The ref and the quota already paid for |
| A subagent re-exploring what its parent already read | **Subagent handoff**: a snapshot of the parent's findings behind one `[handoff hof_…]` stub, and a capped summary back | No repeated exploration |
| Fetching library docs from the web each time | **Doc cache**: `search_docs` / `fetch_doc` serve docs from a local full-text index | No repeated fetches |
| Pasting raw search JSON into the chat | **Code mode**: `execute_code` runs a script over the results and returns only its output | Only the answer |

For example, `internal/selfupdate/selfupdate.go` in this repo costs 1,293 tokens read as a file and 393 as a skeleton.

Every search response reports what it saved (`tokens_saved`, `tokens_used`, and the full-source baseline), and the dashboard totals it, so savings are measured rather than claimed:

```
tokens_saved = max(0, full_source_baseline − tokens_returned) + dedup_skips
```

## Core features

| | |
|--|--|
| **Token-efficient code search** | `get_context_capsule`, `search_semantic`, and `get_file_context` over a symbol-level tree-sitter index; modes `auto` / `skeleton` / `summary` / `full`; per-session dedup; token budgets; filters by path, language, and symbol kind |
| **Virtual context** | `store_context` / `fetch_context` / `search_context` / `edit_context` / `flush_context`: notes, plans, and diffs kept on disk behind stable `ctx_*` refs, so they survive host compaction. Manage them as files — `edit_context` appends, replaces, deletes, or rewrites a note in place and reverts by revision, so an update keeps the ref instead of forcing a re-store and another compaction ([docs](docs/context-edit.md)) |
| **Subagent handoff** | `handoff` / `open_handoff` / `scratchpad`: snapshot-backed briefs for subagents, capped return summaries, and a shared scratchpad with advisory claims for up to 16 parallel children; trees shown on the dashboard ([docs](docs/handoff.md)) |
| **Structured memory** | `store_memory` / `recall_memory` / `forget_memory`: compact temporal facts and procedural rules (`mem_*`), cheaper to recall than notes |
| **Impact and safety checks** | `get_impact_graph`, `diff_impact` (a branch or a GitHub PR), `check_symbol_exists`, `check_deletion_safety`: blast radius before an edit |
| **RAG `retrieve`** | Hybrid search, reranking, and assembly across code, cached docs, and memory in one call |
| **Doc cache** | `fetch_doc` / `search_docs`: library and framework docs cached locally and refreshed weekly |

Around those:

- **Agent installer**: `ast-mcp install` or the dashboard registers the server, skills, rules, and optional hooks for 7 hosts (Claude Code, Cursor, OpenCode, Codex, Claude Desktop, VS Code, JetBrains), with a preview diff, backups, and merge-only edits.
- **Operator dashboard** on port **7830**: tokens saved, index and embedding health, memory, handoff trees, settings, and one-click updates from GitHub releases.
- **Local-first**: SQLite under `~/.astcache/`, local ONNX embeddings or your own backend (Ollama, OpenAI-compatible, Docker Model Runner).

## Quick Start

### From a release

Every merge to `main` publishes a [release](https://github.com/coma-toast/ast-context-cache/releases) for macOS (Apple Silicon) and Linux (arm64):

```bash
tar -xzf ast-context-cache_<version>_darwin_arm64.tar.gz
./ast-mcp --version
```

The local ONNX embedder also needs onnxruntime (`brew install onnxruntime`) and the model files; [`docs/INSTALL.md`](docs/INSTALL.md#install-from-a-release) has the details. Later releases install from the dashboard: Settings → **Updates** downloads the release, checks it against `checksums.txt`, and swaps it in.

### From source

**Prerequisites:** Go 1.25+ with CGO enabled, and Homebrew on macOS (for onnxruntime).

```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
make setup
make run
```

```
MCP: http://127.0.0.1:7821/mcp
Dashboard: http://127.0.0.1:7830
```

`make setup` installs onnxruntime, downloads the embedding model and tokenizer library, and builds the binary (including the React dashboard). Both servers listen on `127.0.0.1` and `[::1]` only; to reach them from another machine, see [remote access](docs/INSTALL.md#remote-access-tailscale). Other embedding backends: [`docs/embedding-backends.md`](docs/embedding-backends.md).

## Configure your editor

Use the installer. It previews a per-file diff, backs up every file it touches, and edits only its own entries, so your other servers, comments, and instructions stay intact:

```bash
ast-mcp install --target cursor --dry-run     # preview
ast-mcp install --target cursor --yes         # apply
ast-mcp verify                                # status for every host
```

Targets: `claude_code`, `cursor`, `opencode`, `codex`, `claude_desktop`, `vscode`, `jetbrains` (or `all`). Components: `mcp`, `skills`, `rules`, `hooks` (opt-in Claude Code subagent-handoff hooks, behind the `feature_handoff_hooks` flag; see [`docs/handoff.md`](docs/handoff.md#claude-code-hooks)). The dashboard offers the same under Settings → **Agent integration**. CLI reference, exit codes, and backups: [`docs/INSTALL.md`](docs/INSTALL.md#connect-your-agents). Exact files and keys per host: [`docs/host-integration.md`](docs/host-integration.md).

`ast-mcp` here is the [shell function](#shell-function-optional) from `make install`; `./ast-mcp` takes the same subcommands.

Manual config is a URL entry, for example Cursor's `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ast-context-cache": { "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

Don't add an `env` block to a URL entry; it has no effect. Set `AST_MCP_TIER` and other server settings on the `ast-mcp` process.

**Agents:** the workflow, tool tiers, virtual context, handoff, and token tips are in **[`AGENTS.md`](AGENTS.md)** and [`skills/usage/SKILL.md`](skills/usage/SKILL.md). The installer ships the same guidance to each host as skills and an instruction block.

## More features

| | |
|--|--|
| **Languages** | Python, JS/JSX, TS/TSX, Go, Bash, Fish, YAML |
| **File watcher** | Incremental re-index with debounce and ignore globs: one FSEvents stream per project on macOS (no descriptor per file), `fsnotify` elsewhere |
| **Embedding backends** | ONNX, Ollama, HTTP, OpenAI/LiteLLM, Docker Model Runner, plus an auxiliary catch-up pool when the primary is down or slow ([docs](docs/embedding-backends.md)) |
| **Context functions** | `define_context_fn` / `apply_context_fn` / `list_context_fns`: reusable transforms over stored notes, off by default ([docs](docs/context-fn.md)) |
| **KV repair** | `report_kv_repair_event`, golden-text archives, success-rate stats on the dashboard |
| **Tool tiers** | `core` / `extended` / `complete`, plus per-tool overrides in `~/.astcache/tools.json` |
| **Feature flags** | Live switches (Settings → Features, `AST_FEATURE_*` env locks); clients are told with `tools/list_changed` |
| **Code mode** | `execute_code` sandbox and repo scripts in `scripts/code-mode/` |
| **Pinning and queue** | Bounded embed queue; pinned projects get priority |
| **Analysis and bundles** | Dead code, complexity, `.astbundle` export/import |
| **Remote access** | Extra listen addresses, trusted hosts, and an optional access token, applied live ([docs](docs/INSTALL.md#remote-access-tailscale)) |
| **Supervise / Docker** | `ast-mcp supervise` or [`docker/ast-mcp`](docker/ast-mcp/README.md) |
| **Metrics** | Prometheus at `http://127.0.0.1:7830/metrics` (`astcache_` prefix) |
| **Structured logs** | `log/slog`; `AST_LOG_FORMAT=text\|json`, `AST_LOG_LEVEL` |

## Screenshots

![Index & runtime](docs/images/dashboard-overview-index-runtime.png)

![Memory](docs/images/dashboard-memory.png)

![Embeddings](docs/images/dashboard-embeddings.png)

![Settings](docs/images/dashboard-settings.png)

Screenshots come from Storybook fixtures, so no live index is needed:

```bash
make dashboard-screenshot   # build ui/ Storybook → docs/images/*.png
make verify-stories         # Playwright smoke test of key stories
```

Storybook: `make storybook` (port **6008**).

## Shell function (optional)

```bash
make install
```

```bash
ast-mcp start | supervise | stop | restart | status | health | log | build | dash
```

The function also passes `install`, `uninstall`, `verify`, `backups`, `restore`, `version`, and `hook` to the binary, so `ast-mcp install --target cursor --dry-run` works from any directory.

**Docker keep-alive:** `docker compose -f docker/ast-mcp/compose.yml up -d --build`. See [`docker/ast-mcp/README.md`](docker/ast-mcp/README.md).

For a local supervisor that runs several MCP servers and merges their host config, see **[mcp-local](https://github.com/coma-toast/mcp-local)**.

## Tool tiers and per-tool overrides

| Tier | Typical tools |
|------|----------------|
| **core** | Search, maps, docs, `retrieve`, impact checks (`get_impact_graph`, `diff_impact`, `check_symbol_exists`, `check_deletion_safety`), context **read**, `recall_memory`, and the handoff tools `handoff` / `open_handoff` / `scratchpad` |
| **extended** | + indexing, `store_context` / `edit_context` / memory write, analysis, bundles, doc-source management (`define_context_fn` / `apply_context_fn` are extended too, behind `feature_context_fn`) |
| **complete** | + `execute_code` |

The handoff tools are core even though they write, so delegation works at every tier; turn them off with a feature flag instead.

`AST_MCP_TIER` (default `complete`), `AST_MCP_CODE_MODE`, and `AST_MCP_TOOLS_CONFIG` / `~/.astcache/tools.json` are read at startup. Feature flags apply live. A tool is listed only when its flag, `tools.json`, and the tier all allow it. Full tables and examples: [AGENTS.md](AGENTS.md#tool-tiers-server-policy), [`skills/tools.json.example`](skills/tools.json.example), [`docs/handoff.md`](docs/handoff.md#feature-flags).

## Upgrading

Settings → **Updates** installs the latest release in place and keeps the previous binary as `ast-mcp.prev`; **Restart now** switches over. From a source checkout, `git pull && make build`, then restart. Breaking changes and what to do about them, including the 3.0 and 4.0 upgrades: [`docs/MIGRATING.md`](docs/MIGRATING.md).

## Architecture

```
┌──────────────┐   MCP (HTTP)     ┌───────────────────┐
│  AI agents   │ ◄──────────────► │  MCP server :7821 │
│  Claude Code │                  │  tree-sitter AST  │
│  Cursor, …   │                  │  SQLite + FTS5    │
└──────────────┘                  │  embeddings       │
                                  └─────────┬─────────┘
                                            │
                                  ┌─────────┴──────────┐
                                  │  Dashboard :7830   │
                                  │  React + /metrics  │
                                  └────────────────────┘
```

**Databases** (WAL, under `~/.astcache/`): `index.db` (symbols, vectors, edges), `context.db` (docs, virtual context, memory), `usage.db` (queries, sessions, settings).

### Environment (common)

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNXRUNTIME_LIB` | ONNX Runtime library path | Auto-detected |
| `MODEL_DIR` | Model files | `./model` |
| `DB_PATH` | Base path for DBs | `~/.astcache/usage.db` |
| `EMBED_AUX_BACKEND` / `EMBED_AUX_WORKERS` | Aux catch-up pool | `onnx` / `0` |
| `AST_CONTEXT_MAX_*` / `AST_CONTEXT_LIMIT_POLICY` | Virtual context quotas | See AGENTS.md / Settings |
| `AST_LISTEN` | Bind address for MCP and dashboard (`--listen`) | `127.0.0.1` |
| `AST_LISTEN_EXTRA` | Extra IPs to listen on beside loopback, e.g. a Tailscale IP ([remote access](docs/INSTALL.md#remote-access-tailscale)); locks Settings → Network access | Unset |
| `AST_TRUSTED_HOSTS` | Extra hostnames (e.g. MagicDNS) accepted in `Host` / `Origin` | Unset |
| `AST_ACCESS_TOKEN` | Bearer token required from non-loopback clients (dashboard: `/login`) | Unset (no token) |
| `AST_MCP_PORT` / `AST_DASHBOARD_PORT` | MCP and dashboard ports (`--mcp-port` / `--dashboard-port`). `ast-mcp install` and the Claude Code hooks read `AST_MCP_PORT` too | `7821` / `7830` |
| `AST_LOG_FORMAT` / `AST_LOG_LEVEL` | Log format (`text` / `json`) and level | `text` / `info` |
| `AST_MCP_TIER` / `AST_MCP_CODE_MODE` / `AST_MCP_TOOLS_CONFIG` | Tool tier, `execute_code` switch, `tools.json` path | `complete` / on / `~/.astcache/tools.json` |
| `AST_FEATURE_*` | Feature-flag locks (e.g. `AST_FEATURE_HANDOFF=false`) | Unset (dashboard toggle) |
| `AST_HANDOFF_*` | Handoff TTL, caps, summary size ([docs](docs/handoff.md#settings-and-limits)) | See docs |

Embedding backend variables: [`docs/embedding-backends.md`](docs/embedding-backends.md). A non-empty environment variable overrides the matching dashboard setting.

### Linux

Install ONNX Runtime (`libonnxruntime-dev` or [releases](https://github.com/microsoft/onnxruntime/releases)), then `make setup`.

### Cross-platform build

`make download-tokenizer-lib` pulls a prebuilt `libtokenizers.a` for your `GOOS`/`GOARCH` from [daulet/tokenizers](https://github.com/daulet/tokenizers/releases).

## License

MIT
