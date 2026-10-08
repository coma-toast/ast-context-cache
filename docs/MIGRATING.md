# Migrating

Upgrade notes for releases that change behavior. Releases from 4.0.8 on update in place from the dashboard (Settings → Updates); see [INSTALL.md](INSTALL.md#updating).

## Migrating to 4.0

4.0 adds subagent handoff, live feature flags, and a safe installer, and tightens defaults. Rebuild (`make build`), restart, then re-run the installer.

| Change | What to do |
|--------|------------|
| **Localhost bind.** MCP and dashboard listen on `127.0.0.1` (plus `[::1]`) instead of every interface. | Nothing for local use. For LAN or container access start with `--listen 0.0.0.0` or `AST_LISTEN=0.0.0.0` (Docker images already set it). Anything that reached the server by LAN IP must be pointed at the new address. |
| **Origin / Host guard.** `/mcp`, dashboard writes, and the WebSocket return **403** for a non-loopback browser `Origin` (`forbidden origin`) or `Host` (`forbidden host`). | Native MCP clients send no `Origin` and are unaffected. Open browser tools from `localhost`; for a deliberate non-loopback setup, listen on the address clients use. |
| **Structured logs.** All logging moved to `log/slog` with `key=value` fields. | Update log scrapers. `AST_LOG_FORMAT=json` gives JSON lines; `AST_LOG_LEVEL=debug\|info\|warn\|error` (default `info`). |
| **Error strings.** Errors use the `errs` package: lowercase messages with structured fields, and handoff tools return `{"error": <code>, "message", "details", "suggestions"}`. | Match on stable codes (`error` field) rather than message text. |
| **MCP protocol.** The server is dual-era: stateless `2026-07-28` (per-request `_meta`, `subscriptions/listen`) plus legacy `initialize` for `2025-11-25`, `2025-06-18`, `2025-03-26`, and `2024-11-05`. It declares `listChanged` and sends `notifications/tools/list_changed`. | Nothing; clients negotiate. Clients that honor `list_changed` refresh their tool list when a flag changes. |
| **Three new core tools**: `handoff`, `open_handoff`, `scratchpad`, gated by flags `feature_handoff`, `feature_handoff_scratchpad`, `feature_handoff_claims`, `feature_handoff_live_trail` (all on), `feature_handoff_hooks` (off). | They appear in `tools/list` at every tier. To hide them, turn the flag off in Settings → Features or set `AST_FEATURE_HANDOFF=false` (an env value locks the flag), or disable them in `tools.json`. See [`docs/handoff.md`](handoff.md). |
| **Claude Code handoff hooks (opt-in).** `SessionStart`, `SubagentStart`, `SubagentStop`, and `PreToolUse` (`Agent`) hooks automate handoffs for Claude Code subagents. | Optional. Turn on `feature_handoff_hooks`, then `ast-mcp install --target claude_code --component hooks --yes` and restart Claude Code. They fail open. See [`docs/handoff.md`](handoff.md#claude-code-hooks). |
| **Query cache with `session_id`.** The search cache is now shared across sessions and used for `session_id` calls too (`feature_shared_query_cache`); entries are invalidated when covered files are reindexed. Dedup is immediate: a symbol returned to a session is deduped on its very next call (previously up to ~3s later). | Nothing. Results per session are unchanged; cache-hit statistics and latency improve. Set `AST_FEATURE_SHARED_QUERY_CACHE=false` to opt out. |
| **Scope fixes.** `search_context`'s keyword fallback now honors `session_id` / `project_path`, and memory vector recall no longer returns superseded, forgotten, or out-of-scope entries. | Nothing; results that leaked across scopes disappear. |
| **`retrieve` stats.** `stats.deduped_count` is now `stats.deduped`. | Rename the field in scripts that read it. |
| **Installer API.** `/api/agent-*` and project-scope installs are gone. | Use `ast-mcp install\|uninstall\|verify\|backups\|restore` ([`docs/INSTALL.md`](INSTALL.md#connect-your-agents)) or `/api/dashboard/installer*` (Settings → Agent integration). |
| **Pre-4.0 installs.** The old installer overwrote whole files; its Claude Code target could write markdown over `~/.claude.json`. | Re-run `ast-mcp install --target <host> --yes` (or `--target all`). The first 4.0 start re-checks old install records and shows legacy warnings in `verify` and the dashboard. If Claude Code lost its config, restore `~/.claude.json` from `~/.claude/backups/` first. |
| **Claude Desktop** needs a stdio bridge: the installer writes `mcp-local bridge <url>`, or `npx -y mcp-remote <url>` when mcp-local is not on `PATH`. | Install the [mcp-local](https://github.com/coma-toast/mcp-local) release that adds the `bridge` command (the companion of this release), or have Node.js for `npx`, then re-run `ast-mcp install --target claude_desktop --yes`. |
| **No `env` on URL entries.** Docs no longer suggest `env` blocks on `url` server entries; they never had an effect. | Remove them from your host configs if present, and set `AST_MCP_TIER` on the `ast-mcp` process instead. |

## Migrating to 3.0

| Change | What to do |
|--------|------------|
| Removed MCP ghost tools | `sync_remote` / `reset_*` gone from MCP; use local index + dashboard APIs |
| React-only dashboard | SPA at `http://localhost:7830/dashboard/` — not HTMX/templ |
| Keep-alive | `ast-mcp supervise` or Docker Compose `restart: unless-stopped` |
| Prometheus | Scrape `http://localhost:7830/metrics` |
| Overview confidence | Heuristics, weekly digest, session virtual-context stories |

Rebuild after upgrading (`make build` / `ast-mcp build`).
