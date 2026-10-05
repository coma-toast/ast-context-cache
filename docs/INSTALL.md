# Installing ast-context-cache

## Prerequisites

- Go 1.25+ with CGO enabled
- macOS (with Homebrew) or Linux
- onnxruntime library (installed by `make setup` / `make deps`)

## Quick install

```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
make setup
make run
```

`make setup` installs the dependencies and builds the `ast-mcp` binary in the repo root. Then [connect your agents](#connect-your-agents).

## Manual steps

### 1. Install dependencies

```bash
make deps
```

This installs onnxruntime (via brew on macOS), downloads the embedding model, and downloads the tokenizer library.

### 2. Build

```bash
make build
```

### 3. Run

```bash
make run
```

Starts the MCP server on `http://127.0.0.1:7821/mcp` and the dashboard on `http://127.0.0.1:7830`.

| Flag / env | Default | Purpose |
|---|---|---|
| `--mcp-port` / `AST_MCP_PORT` | 7821 | MCP HTTP port. The installer CLI (`--mcp-port` default) and the Claude Code hooks read `AST_MCP_PORT` too, so one export moves the server, its registrations, and the hooks together. |
| `--dashboard-port` / `AST_DASHBOARD_PORT` | 7830 | Dashboard HTTP port |
| `--listen` / `AST_LISTEN` | `127.0.0.1` | Address both servers bind. The default loopback bind also listens on `[::1]`, so `http://localhost:7821/mcp` works wherever `localhost` resolves to IPv6. Docker sets `0.0.0.0` so published ports work. |
| `AST_LOG_FORMAT` | `text` | `text` or `json` (structured `slog` output) |
| `AST_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, or `error` |

**Network exposure.** The servers have no authentication. Binding anything other than loopback (for example `--listen 0.0.0.0`) makes indexed code, virtual context, and handoff data readable from that network, and the server logs a warning. Keep the default unless you need remote access.

**Origin and Host checks.** `/mcp`, the dashboard's mutating routes, and its WebSocket reject a browser `Origin` that is not loopback (`localhost`, `127.0.0.1`, `[::1]`, any port) with **403 `forbidden origin`**, and a non-loopback `Host` header with **403 `forbidden host`** (unless the server listens on a wildcard address). Native MCP clients and `curl` send no `Origin` and are unaffected. If a browser-based tool gets 403s, open it from `localhost`.

## Shell function (optional)

```bash
make install
```

Adds an `ast-mcp start|start-safe|supervise|stop|restart|status|health|log|build|dash` shell function (bash, zsh, fish). It also passes `install`, `uninstall`, `verify`, `backups`, `restore`, `version` (`--version`), and `hook` straight to the built binary, with their exit codes, so the [installer commands](#cli) below work from any directory. The function and the server both follow `AST_MCP_PORT` / `AST_DASHBOARD_PORT`.

## Connect your agents

The installer registers the MCP server and installs skills, rules or instruction blocks, and (opt-in) Claude Code hooks for **Claude Code, Cursor, OpenCode, Codex, Claude Desktop, VS Code, and JetBrains**. It is merge-only: it previews a per-file diff, backs up every file it changes, edits only its own entries and marker blocks, and preserves comments in JSONC and TOML. The exact files and keys per host are in [host-integration.md](host-integration.md).

### Dashboard

Settings → **Agent integration**: pick targets and components, **Preview** the diff, then **Apply**. Backups are listed with a Restore button. It registers the running server's actual MCP port.

### CLI

The CLI runs in-process against the same installer; the server does not need to be running. `ast-mcp` below is the [shell function](#shell-function-optional); without it, run the binary directly (`./ast-mcp` in the repo), which takes the same arguments.

```bash
ast-mcp install --target cursor --dry-run          # preview the diff, write nothing
ast-mcp install --target cursor --yes              # apply
ast-mcp install --target claude_code,codex --component mcp,rules --yes
ast-mcp install --target all --yes                 # every target; unsupported components are skipped
ast-mcp verify                                     # status table for every target
ast-mcp uninstall --target cursor --yes            # remove only what the installer added
ast-mcp backups                                    # list backups (newest first)
ast-mcp restore --yes 20261005-142233/%Users%me%.cursor%mcp.json
ast-mcp version
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--target` | install, uninstall, verify | `claude_code`, `cursor`, `opencode`, `codex`, `claude_desktop`, `vscode`, `jetbrains`, or `all`. Repeatable or comma-separated. Required for install/uninstall; verify defaults to all. |
| `--component` | install, uninstall | `mcp`, `skills`, `rules`, `hooks` (comma list). Default: every supported component. `hooks` (Claude Code only, opt-in) is offered only when the `feature_handoff_hooks` flag is on; see [handoff.md](handoff.md#claude-code-hooks). |
| `--dry-run` | install, uninstall | Print the diff and statuses without writing. |
| `--yes` | install, uninstall, restore | Apply. Without `--yes` or `--dry-run`, install/uninstall print the preview and exit 2. |
| `--mcp-url` | install, uninstall, verify | Register this URL verbatim. |
| `--mcp-port` | install, uninstall, verify | Register `http://127.0.0.1:<port>/mcp`. Default `$AST_MCP_PORT`, then 7821. Use it when the server runs with a non-default `--mcp-port`. |
| `--replace-external` | install, uninstall | Replace externally managed skill or rule paths (for example a symlinked `~/.claude/skills/ast-context-cache`); a backup is taken first. |
| `--json` | all | Machine-readable output: `{"changes": [...], "status": [...], "warnings": [...]}` (`backups` prints the backup list). |

| Exit code | Meaning |
|---|---|
| 0 | OK (applied, previewed with `--dry-run`, or nothing to do) |
| 1 | Error (bad flags, unknown target, database unavailable) |
| 2 | Confirmation required: changes are pending; re-run with `--yes` |
| 3 | Conflict: a file changed since the preview, or a config file could not be parsed (nothing was written) |
| 4 | Unsupported: every requested component is unsupported for a named target |

`ast-mcp hook <event>` is the command the installed Claude Code hooks run; you don't call it yourself except to debug ([handoff.md](handoff.md#claude-code-hooks)).

**Backups** live in `~/.astcache/backups/<YYYYMMDD-HHMMSS>/`, one file per modified path with `/` replaced by `%`. The newest 5 per file are kept (setting `installer_backup_keep`). `restore` backs up the current file before writing the old one back.

**Status values** reported by `verify` and the dashboard: `installed`, `outdated`, `modified_by_user`, `missing`, `not_installed`, `externally_managed`, `covered`, `unsupported`. See [host-integration.md](host-integration.md#status-values).

**Upgrading from 3.x.** Re-run the installer to repair pre-4.0 installs. The first 4.0 start re-checks the old install records and prints warnings (also in `verify`); in particular, the old Claude Code target could overwrite `~/.claude.json`, which Claude Code backs up in `~/.claude/backups/`. Close Claude Code before applying a change to `~/.claude.json`, because it rewrites that file while running.

### Manual registration

If you prefer to edit config yourself, use the same entries the installer writes. Never add an `env` block to a URL entry: hosts apply `env` only to processes they launch, so it has no effect on an HTTP server. Set `AST_MCP_TIER` and friends on the `ast-mcp` process instead.

| Host | File | Entry |
|---|---|---|
| Claude Code | `~/.claude.json` (or `claude mcp add --transport http --scope user ast-context-cache http://127.0.0.1:7821/mcp`) | `"mcpServers": {"ast-context-cache": {"type": "http", "url": "http://127.0.0.1:7821/mcp"}}` |
| Cursor | `~/.cursor/mcp.json` | `"mcpServers": {"ast-context-cache": {"url": "http://127.0.0.1:7821/mcp"}}` |
| OpenCode | `~/.config/opencode/opencode.jsonc` (or `.json`) | `"mcp": {"ast-context-cache": {"type": "remote", "url": "http://127.0.0.1:7821/mcp", "enabled": true}}` |
| Codex | `~/.codex/config.toml` | `[mcp_servers.ast-context-cache]` then `url = "http://127.0.0.1:7821/mcp"` |
| VS Code | user `mcp.json` ("MCP: Open User Configuration") | `"servers": {"ast-context-cache": {"type": "http", "url": "http://127.0.0.1:7821/mcp"}}` |
| Claude Desktop | `claude_desktop_config.json` | `"mcpServers": {"ast-context-cache": {"command": "/abs/path/to/mcp-local", "args": ["bridge", "http://127.0.0.1:7821/mcp"]}}` (needs [mcp-local](https://github.com/coma-toast/mcp-local) with the `bridge` command), or `{"command": "npx", "args": ["-y", "mcp-remote", "http://127.0.0.1:7821/mcp"]}` |
| JetBrains AI Assistant | Settings → Tools → AI Assistant → Model Context Protocol (MCP) → Add, server level Global | `{"mcpServers": {"ast-context-cache": {"url": "http://127.0.0.1:7821/mcp"}}}` |

For agent instructions, paste [`instructions/agents-block.md`](../instructions/agents-block.md) into the host's global instructions file (for example `~/.claude/CLAUDE.md` or `~/.codex/AGENTS.md`), or for Cursor use [`rules/cursor/ast-context-cache.mdc`](../rules/cursor/ast-context-cache.mdc).

## Tool tiers and feature flags

`AST_MCP_TIER` (`core` / `extended` / `complete`, default `complete`), `AST_MCP_CODE_MODE`, and `tools.json` (`AST_MCP_TOOLS_CONFIG`, default `~/.astcache/tools.json`) decide which tools agents see; they are read at server start. Feature flags (`AST_FEATURE_*`, or dashboard Settings → Features) switch whole features such as subagent handoff on and off live. See [README](../README.md#tool-tiers-and-per-tool-overrides) and [handoff.md](handoff.md#feature-flags).

## Dashboard

Visit `http://127.0.0.1:7830` for query statistics and token savings, index health, embeddings, memory and virtual context, handoff trees, settings (feature flags, agent integration), and Prometheus metrics at `/metrics`.

## Troubleshooting

### "library 'tokenizers' not found" or `Undefined symbols ... tokenizers_version`

Usually a missing or **wrong-architecture** `libtokenizers.a` (e.g. arm64 lib on an Intel Mac). `make build` re-downloads when arch mismatches (`darwin-amd64` uses the `darwin-x86_64` release asset). To force:

```bash
make clean-tokenizer-lib download-tokenizer-lib
make build
```

### Model files missing

```bash
make download-model
```

### Port already in use

```bash
lsof -i :7821
# Kill the existing process, or move to another port: export AST_MCP_PORT=<port> before starting the
# server (ast-mcp install and the Claude Code hooks read it too), or pass --mcp-port to both.
```

### 403 `forbidden origin` / `forbidden host`

A browser page or proxy reached the server with a non-loopback `Origin` or `Host`. Use `http://localhost:<port>` or `http://127.0.0.1:<port>`. For deliberate LAN access, start with `--listen` set to the address clients use.

### The installer refuses to edit a file

A parse error (exit 3) means the existing config is not valid JSON, JSONC, or TOML; fix it and re-run. VS Code tolerates some invalid JSON that the installer will not rewrite; its error message includes the manual steps.
