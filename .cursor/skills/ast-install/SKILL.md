---
name: ast-context-cache-install
description: Use when installing, configuring, or troubleshooting ast-mcp, registering it with editors (ast-mcp install / verify / uninstall), tool tiers (AST_MCP_TIER for store_context/flush_context), or tools.json overrides.
---

## When to Use

When you need to install or setup ast-context-cache for an AI coding agent.
Use this skill when the user asks to:
- Install ast-context-cache
- Setup MCP server for their agent
- Configure Cursor, OpenCode, Claude, etc.
- Troubleshoot installation issues
- Verify, repair, or remove an editor integration (`ast-mcp install` / `verify` / `uninstall`)

## Installation Steps

### 1. Clone and Setup
```bash
git clone https://github.com/coma-toast/ast-context-cache.git
cd ast-context-cache
make setup
```

### 2. Start Server
```bash
make run
```
Server runs at `http://127.0.0.1:7821/mcp`, dashboard at `http://127.0.0.1:7830`. Both bind loopback only; use `--listen` / `AST_LISTEN` (for example `0.0.0.0` in Docker) to expose them, knowing there is no authentication.

### 3. Connect Your Agents (installer)

Use the installer instead of hand-editing config. It previews a per-file diff, backs up each file before writing, merges only its own entry or marker block (comments in JSONC/TOML survive), and never deletes files it did not create.

```bash
ast-mcp install --target cursor --dry-run        # preview only
ast-mcp install --target cursor --yes            # apply
ast-mcp install --target all --yes               # every host; unsupported parts are skipped
ast-mcp verify                                   # installed / outdated / modified_by_user / …
ast-mcp uninstall --target cursor --yes          # remove only what was added
ast-mcp backups && ast-mcp restore --yes <id>    # undo a change
```

- Targets: `claude_code`, `cursor`, `opencode`, `codex`, `claude_desktop`, `vscode`, `jetbrains`. Components: `mcp`, `skills`, `rules`, `hooks` (Claude Code hooks only appear when the `feature_handoff_hooks` flag is on).
- `--mcp-port` / `--mcp-url` when the server is not on 7821 (default `$AST_MCP_PORT`, then 7821). `--json` for scripts.
- Exit codes: 0 ok, 1 error, 2 needs `--yes`, 3 conflict or unparseable config (nothing written), 4 unsupported.
- `ast-mcp` here is the shell function from `make install`, which passes these subcommands to the built binary; `./ast-mcp` in the repo works the same.
- Same flow in the dashboard: Settings → **Agent integration** (Preview, Apply, Backups).
- Upgrading from 3.x: re-run install. The old Claude Code installer could overwrite `~/.claude.json`; restore it from `~/.claude/backups/` if needed. Close Claude Code before applying changes to `~/.claude.json`.

Reference: [docs/INSTALL.md](../../../docs/INSTALL.md#connect-your-agents) (flags, manual snippets); files and keys per host: [docs/host-integration.md](../../../docs/host-integration.md).

**Manual config:** use the exact entries in [agents/SKILL.md](../../../skills/agents/SKILL.md#editor-mcp-configuration-manual). **Never add an `env` block to a `url` entry** — hosts apply `env` only to processes they launch, so it does nothing for an HTTP server. Set `AST_MCP_TIER` on the `ast-mcp` process instead.

See [usage/SKILL.md](../ast-usage/SKILL.md) for full workflows including virtual context compaction and subagent handoff.

### 4. Agent instructions (all editors)

The installer's `skills` and `rules` components install the canonical skills and instruction block for each host (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.config/opencode/AGENTS.md`, `~/.cursor/rules/ast-context-cache.mdc`). For a project `AGENTS.md` / `CLAUDE.md`, paste the block from [agents/SKILL.md](../../../skills/agents/SKILL.md#agent-instructions-block-paste-into-agentsmd-or-claudemd).

## Tool tiers

| Tier | Virtual context | Handoff |
|------|-----------------|---------|
| core | `fetch_context`, `list_context`, `search_context` | `handoff`, `open_handoff`, `scratchpad` |
| extended | + `store_context`, `flush_context` | (same) |

Default server tier is `complete`. For read-only agents: `AST_MCP_TIER=core` (the handoff tools still appear; turn them off with `AST_FEATURE_HANDOFF=false` or Settings → Features). For indexing without code mode: `AST_MCP_TIER=extended`.

Per-tool overrides: `~/.astcache/tools.json` (or `AST_MCP_TOOLS_CONFIG`) — see [tools.json.example](../../../skills/tools.json.example). Restart ast-mcp after edits. Feature flags apply live, no restart.

## Troubleshooting

### "library 'tokenizers' not found"
```bash
make download-tokenizer-lib
```

### Model files missing
```bash
make download-model
```

### Port already in use
```bash
lsof -i :7821
```

### 403 `forbidden origin` / `forbidden host`
A browser tool or proxy reached the server with a non-loopback `Origin` or `Host`. Use `http://localhost:<port>`; for deliberate LAN access start ast-mcp with `--listen <address>`.

### Installer exits 3
The target config failed to parse (nothing was written) or changed since the preview. Fix the file or re-run the command.

## Shell Function (Optional)
```bash
make install
```
Adds `ast-mcp start|stop|restart|status|health` to your shell.
