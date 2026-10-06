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
| `AST_LISTEN_EXTRA` | unset | Extra IPs both servers also listen on, comma or newline separated (for example a Tailscale IP). Overrides and locks Settings → Network access. See [Remote access](#remote-access-tailscale). |
| `AST_TRUSTED_HOSTS` | unset | Extra hostnames (for example a MagicDNS name) accepted in `Host` and `Origin` headers. Overrides and locks the setting. |
| `AST_ACCESS_TOKEN` | unset | Token every non-loopback client must present. Overrides and locks the setting. |
| `AST_LOG_FORMAT` | `text` | `text` or `json` (structured `slog` output) |
| `AST_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, or `error` |

**Network exposure.** Loopback clients never authenticate. Anything else that can reach the servers can read indexed code, virtual context, and handoff data unless you set an access token, so keep the default bind unless you need remote access, and prefer [extra listen addresses plus a token](#remote-access-tailscale) over `--listen 0.0.0.0`. The server logs a warning for a non-loopback bind and for extra addresses without a token.

**Origin and Host checks.** `/mcp`, the dashboard's mutating routes, and its WebSocket reject a browser `Origin` that is not loopback (`localhost`, `127.0.0.1`, `[::1]`, any port) or a trusted host with **403 `forbidden origin`**, and any other `Host` header with **403 `forbidden host`** (unless the server listens on a wildcard address). Trusted hosts are the extra listen addresses plus the trusted hostnames from Settings → Network access. Native MCP clients and `curl` send no `Origin` and are unaffected. If a browser-based tool gets 403s, open it from `localhost` or add its hostname to trusted hosts.

## Remote access (Tailscale)

To use the server from another machine on your tailnet, keep the loopback bind and add the Tailscale address on top. Everything below applies live; no restart.

1. **Listen on the Tailscale IP.** Run `tailscale ip -4` on the server machine and add the address under dashboard **Settings → Network access → Extra listen addresses** (or set `AST_LISTEN_EXTRA=100.x.y.z`). Both servers keep listening on `127.0.0.1` and `[::1]`, and also open `100.x.y.z:7821` and `100.x.y.z:7830`. The status chips show each listener; one that fails (the address isn't on this host yet, the port is taken) is retried every 30 seconds, so it comes up once `tailscaled` does.
2. **Trust the MagicDNS name.** Add the machine's MagicDNS name (for example `my-laptop.tailnet-name.ts.net`, and the short name `my-laptop` if you use it) under **Trusted hostnames** (`AST_TRUSTED_HOSTS`). Extra listen addresses are trusted automatically; hostnames have to be listed so the dashboard opened at `http://my-laptop.tailnet-name.ts.net:7830` can save settings.
3. **Set an access token.** Click **Generate** under **Access token** (or set `AST_ACCESS_TOKEN`), and copy the token: it is shown once and never returned by the API. With a token set, every request that does not come from loopback must carry `Authorization: Bearer <token>`, on both ports and every method, WebSocket upgrades included. Remote browsers are sent to `/login` once, which sets an `HttpOnly`, `SameSite=Strict` session cookie derived from the token; generating a new token signs everyone out. The MCP port's `/health` stays open for monitors. Without a token the dashboard warns that anyone who can reach the address can use the server.
4. **Point the remote client at it**, with the token in a header (never in the URL):

   Claude Code (`~/.claude.json`, or `claude mcp add --transport http --scope user ast-context-cache http://100.x.y.z:7821/mcp --header "Authorization: Bearer <token>"`):

   ```json
   "mcpServers": {
     "ast-context-cache": {
       "type": "http",
       "url": "http://100.x.y.z:7821/mcp",
       "headers": { "Authorization": "Bearer <token>" }
     }
   }
   ```

   Cursor (`~/.cursor/mcp.json`):

   ```json
   "mcpServers": {
     "ast-context-cache": {
       "url": "http://my-laptop.tailnet-name.ts.net:7821/mcp",
       "headers": { "Authorization": "Bearer <token>" }
     }
   }
   ```

   Other hosts take the same URL plus an `Authorization` header in their HTTP server config. A wrong or missing token gets **401** with JSON-RPC error code `-32001` (`unauthorized`).

Clients on the server machine itself (including the Claude Code hooks and the installer) keep using `http://127.0.0.1:7821/mcp` with no token. Use [Tailscale ACLs](https://tailscale.com/kb/1018/acls) to limit which devices can reach ports 7821 and 7830 at all, as defense in depth.

Notes:

- Traffic is plain HTTP inside the tailnet (WireGuard encrypts it between devices). Don't expose these ports beyond networks you trust.
- A reverse proxy on the same machine, such as `tailscale serve`, connects from loopback, so its clients skip the token. Put authentication in the proxy if you use one.
- With `--listen 0.0.0.0` (Docker) extra addresses are not opened separately because the base bind already covers every interface; they still count as trusted hosts. Set `AST_ACCESS_TOKEN` in the environment there, since a token stored in Settings only takes effect once the databases have opened during startup.

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

A browser page or proxy reached the server with an `Origin` or `Host` that is neither loopback nor trusted. Use `http://localhost:<port>` or `http://127.0.0.1:<port>`, or for deliberate remote access add the address to Settings → Network access (extra listen addresses or trusted hostnames; see [Remote access](#remote-access-tailscale)).

### 401 `unauthorized`

An access token is set and the request came from another machine without `Authorization: Bearer <token>` or a dashboard session. Add the header to the MCP client config, or sign in at `http://<address>:7830/login`. After the token is regenerated, update every remote client.

### The installer refuses to edit a file

A parse error (exit 3) means the existing config is not valid JSON, JSONC, or TOML; fix it and re-run. VS Code tolerates some invalid JSON that the installer will not rewrite; its error message includes the manual steps.
