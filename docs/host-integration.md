# Host integration inventory (IN-14)

For each agent host the v4.0 installer supports, this document lists exactly which files and keys it writes per component, how it reports status, and the host research behind each choice. The installer lives in `internal/installer`; run it with `ast-mcp install` (see [INSTALL.md](INSTALL.md#connect-your-agents)) or from the dashboard (Settings → Agent integration).

- **Targets:** `claude_code`, `cursor`, `opencode`, `codex`, `claude_desktop`, `vscode`, `jetbrains`. Global (user) scope only; project-scope installs were removed in 4.0.
- **Components:** `mcp` (server registration), `skills`, `rules` (rule file or instruction block), `hooks` (Claude Code subagent-handoff hooks; available, opt-in behind the `feature_handoff_hooks` flag).
- **Server name:** `ast-context-cache` everywhere.
- **Server URL:** `http://127.0.0.1:<port>/mcp`. The dashboard uses the running server's MCP port; the CLI uses `--mcp-url`, else `--mcp-port`, else `$AST_MCP_PORT`, else 7821.
- **Platforms:** macOS and Linux. On Windows every component reports `unsupported` ("Windows is not supported yet").
- **No `env` on URL entries (IN-12).** An `env` block on a `url` entry does nothing (env applies only to processes a host launches), so the installer never writes one. Tier and code-mode settings belong on the `ast-mcp` process (`AST_MCP_TIER`, `AST_MCP_CODE_MODE`), not in host config.

## What the installer writes

`~` is the user's home directory. A dash means the component is reported `unsupported` with the reason shown in the preview.

| Target | `mcp` | `skills` | `rules` | `hooks` |
|---|---|---|---|---|
| `claude_code` | `~/.claude.json` → top-level `mcpServers["ast-context-cache"]` = `{"type": "http", "url": …}` | `~/.claude/skills/ast-context-cache-<name>/SKILL.md` | Marker block in `~/.claude/CLAUDE.md` | `~/.claude/settings.json` → `hooks` (only when `feature_handoff_hooks` is on; see [handoff.md](handoff.md#claude-code-hooks)) |
| `cursor` | `~/.cursor/mcp.json` → `mcpServers["ast-context-cache"]` = `{"url": …}` | `~/.agents/skills/ast-context-cache-<name>/SKILL.md`, or `covered` when ours are already in `~/.claude/skills` or `~/.agents/skills` | `~/.cursor/rules/ast-context-cache.mdc` (whole file: `alwaysApply` frontmatter + marker block) | — Cursor hooks are not part of v4 |
| `opencode` | `~/.config/opencode/opencode.jsonc` if it exists, else `opencode.json` → `mcp["ast-context-cache"]` = `{"type": "remote", "url": …, "enabled": true}` | Same as Cursor (`~/.agents/skills`, or `covered`) | Marker block in `~/.config/opencode/AGENTS.md` | — needs a plugin |
| `codex` | `~/.codex/config.toml` → `[mcp_servers.ast-context-cache]` with `url = …`, preceded by `# managed by ast-context-cache v<version>` | `~/.agents/skills/ast-context-cache-<name>/SKILL.md` | Marker block in `~/.codex/AGENTS.md` | — |
| `claude_desktop` | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `~/.config/Claude/claude_desktop_config.json` (Linux) → `mcpServers["ast-context-cache"]` = `{"command": "<abs path>/mcp-local", "args": ["bridge", url]}`, or `{"command": "npx", "args": ["-y", "mcp-remote", url]}` when `mcp-local` is not on `PATH` (with a preview warning, and a second one if `npx` is missing too) | — no documented location | — no documented location | — |
| `vscode` | `~/Library/Application Support/Code/User/mcp.json` (macOS) or `~/.config/Code/User/mcp.json` (Linux) → `servers["ast-context-cache"]` = `{"type": "http", "url": …}` | — user-scope path unverified | — user-scope path unverified | — |
| `jetbrains` | — the reason carries manual steps (AI Assistant has no documented config file) | — | — | — |

The skills are the four canonical skills from [`skills/`](../skills/README.md) (`usage`, `agents`, `install`, `operator`), each written as `ast-context-cache-<name>/SKILL.md` with `name`/`description` frontmatter added. The instruction block is [`instructions/agents-block.md`](../instructions/agents-block.md); the Cursor rule is [`rules/cursor/ast-context-cache.mdc`](../rules/cursor/ast-context-cache.mdc).

### Ownership and safety rules

- **JSON and JSONC** files are edited with a comment-preserving parser (hujson): only our server key is added, replaced, or removed; other servers, comments, and formatting stay as they were. A file that fails to parse aborts that target with an error and nothing is written. VS Code's parse error message includes the manual steps.
- **TOML** (`~/.codex/config.toml`) is validated, then only the `[mcp_servers.ast-context-cache]` table span (and its managed-by comment) is replaced, appended, or removed.
- **Markdown blocks** sit between `<!-- ast-context-cache:begin v=<version> sha=<hash> -->` and `<!-- ast-context-cache:end -->`. Upgrades replace only the text between the markers; text outside them is never touched. A second or malformed begin marker, or a begin without an end, aborts the edit.
- **Skills** are our own `ast-context-cache-<name>` directories. A skill file the installer did not create, a symlinked skill directory resolving outside `~/.astcache`, or the legacy un-suffixed `ast-context-cache` skill directory is `externally_managed` and skipped unless you pass `--replace-external` (dashboard: "replace external"), which backs it up first. A symlink is recorded in the backup and removed; a real directory is left in place with a warning to remove it by hand.
- **The Cursor rule file** is ours as a whole. An existing `~/.cursor/rules/ast-context-cache.mdc` without our markers is `externally_managed` and skipped unless `--replace-external`.
- **Hooks** are appended as separate matcher groups to each event array in `~/.claude/settings.json` (`SessionStart` matcher `startup|resume|compact`, `SubagentStart`, `SubagentStop`, and `PreToolUse` matcher `Agent`, each with a 3-second timeout); your own hooks are never replaced. Ours are recognized by their command (`<abs path>/ast-mcp hook <event>`). Behavior: [handoff.md](handoff.md#claude-code-hooks).
- **Uninstall** removes only our entry, block, hook groups, or skill files. A file, the top-level key holding our entry, or a `hooks` object is deleted only when the installer created it and it is otherwise empty.
- **Backups:** before any write each modified file is copied to `~/.astcache/backups/<YYYYMMDD-HHMMSS>/<path with / replaced by %>`. The newest 5 per file are kept (setting `installer_backup_keep`). List with `ast-mcp backups`; restore with `ast-mcp restore --yes <id>` (the current file is backed up first).
- **Preview first:** a plan is held for 10 minutes. Apply re-hashes every file and fails with a conflict, writing nothing, if any changed since the preview.
- **`~/.claude.json`:** Claude Code rewrites this file while it runs, so the preview warns to apply with Claude Code closed.

### Status values

`ast-mcp verify` and the dashboard compute status from the files on disk plus the installer's state table, never from the database alone.

| Status | Meaning |
|---|---|
| `installed` | The current version of our entry, block, files, or hooks is present. |
| `outdated` | Our content is present as the installer last wrote it (or partially), but this version would write something different. Re-run install. |
| `modified_by_user` | Our content is present but was edited after the installer wrote it. Install replaces it (with a backup and a warning). |
| `missing` | The state table records an install that is no longer on disk. |
| `not_installed` | Nothing of ours is present and nothing is recorded. |
| `externally_managed` | A path we would write belongs to something else (see above). Skipped unless replaced explicitly. |
| `covered` | The host already loads our skills from a directory another target installs into (Cursor and OpenCode read `~/.claude/skills` and `~/.agents/skills`), so a second copy would only duplicate them. |
| `unsupported` | The host has no supported location for the component; the reason says why (and, for JetBrains MCP, how to add it by hand). The `hooks` component is also `unsupported` while `feature_handoff_hooks` is off. |

### Legacy (pre-4.0) installs

The first 4.0 start re-checks every install record the old dashboard installer left (IN-13) and shows warnings in `ast-mcp verify`, `ast-mcp install`, and the dashboard. The old Claude Code global target wrote markdown over `~/.claude.json`; if Claude Code lost its settings, restore `~/.claude.json` from Claude Code's own backups in `~/.claude/backups/`, then run `ast-mcp install --target claude_code`. Old project-scope records are listed so you can remove those files by hand.

---

# Host research notes

Verified 2026-10-05 against each host's documentation, unless marked **UNVERIFIED**. These notes explain the choices above and list locations the installer deliberately does not use.

## Research summary matrix

| Target | MCP registration | Skills | Rules / Instructions | Hooks |
| - | - | - | - | - |
| **Claude Code** | `~/.claude.json` → top-level `mcpServers.<name>` = `{"type":"http","url":…}`. Prefer the CLI: `claude mcp add --transport http --scope user …`. **HTTP native.** | `~/.claude/skills/<name>/SKILL.md` | `~/.claude/CLAUDE.md` and `~/.claude/rules/*.md` | `~/.claude/settings.json` → `"hooks"` |
| **Cursor** | `~/.cursor/mcp.json` → `mcpServers.<name>` = `{"url":…}`. **HTTP native.** | `~/.cursor/skills/<name>/SKILL.md` (also reads `~/.agents/skills`, `~/.claude/skills` and `~/.codex/skills`) | Global "User Rules" live in the Settings UI (Customize → Rules). `~/.cursor/rules` is **not documented**; the installer writes there anyway, with a warning (see [Cursor](#cursor)). | `~/.cursor/hooks.json` (`{"version":1,"hooks":{…}}`) |
| **OpenCode** | `~/.config/opencode/opencode.json` (or `.jsonc`) → `mcp.<name>` = `{"type":"remote","url":…,"enabled":true}`. **HTTP native (remote).** | `~/.config/opencode/skills/<name>/SKILL.md` (also reads `~/.claude/skills` and `~/.agents/skills`) | `~/.config/opencode/AGENTS.md`; extra files via the `"instructions": [...]` key | Plugins only (`~/.config/opencode/plugins/`); no declarative hooks file |
| **Claude Desktop** | `claude_desktop_config.json` → `mcpServers.<name>` = `{"command","args","env"}`. **stdio only, so a bridge is needed.** | Unsupported — no documented local skills folder | Unsupported — no documented global instructions file | Unsupported |
| **Codex CLI** | `~/.codex/config.toml` → `[mcp_servers.<name>]` `url = …`. **Streamable HTTP native**, no flag. | `~/.agents/skills/<name>/SKILL.md` | `~/.codex/AGENTS.md` (or `$CODEX_HOME/AGENTS.md`) | `~/.codex/hooks.json` or `[hooks]` tables in `~/.codex/config.toml` |
| **VS Code (Copilot)** | User-profile `mcp.json` → `servers.<name>` = `{"type":"http","url":…}`. Docs now prefer portable `~/.copilot/mcp-config.json` → `mcpServers`. **HTTP native.** | UNVERIFIED (Agent Skills supported; user path not checked) | `~/.copilot/copilot-instructions.md` and `~/.copilot/instructions/*.instructions.md` (Agent Host). The Local agent uses VS Code profile storage and also reads `~/.claude/CLAUDE.md`. | UNVERIFIED (hooks exist; path not checked) |
| **JetBrains AI Assistant** | **Unsupported (manual steps)** — UI only: Settings → Tools → AI Assistant → Model Context Protocol (MCP). Paste `{"mcpServers":{"ast-context-cache":{"url":…}}}` and choose Global. **HTTP native.** | Unsupported | Unsupported (not researched) | Unsupported |
| **JetBrains Junie** | `~/.junie/mcp/mcp.json` (user scope); `.junie/mcp/mcp.json` (project scope). The IDE's Junie MCP Settings also write `~/.junie/mcp/mcp.json`. Remote entry shape is **UNVERIFIED**. | UNVERIFIED (Junie documents "Agent skills") | UNVERIFIED (Junie documents "Guidelines and memory") | UNVERIFIED (Junie documents "Hooks") |

### Transport quick view

| Target | Remote HTTP / streamable natively? | Bridge needed? |
| - | - | - |
| Claude Code | Yes (`type: "http"`, alias `streamable-http`). Since v2.1.265 it auto-falls back to SSE. The v2 runtime negotiates 2026-07-28. | No |
| Cursor | Yes (stdio, SSE, Streamable HTTP) | No |
| OpenCode | Yes (`type: "remote"`). The docs do not name the HTTP sub-transport, but the user's live config already points `remote` at `localhost:7821/mcp`. | No |
| Claude Desktop | No. `claude_desktop_config.json` is documented with `command`/`args` only. Remote servers go through **Custom Connectors**, which are brokered by claude.ai and cannot reach `127.0.0.1`. | **Yes — stdio bridge** |
| Codex CLI | Yes ("Streamable HTTP servers: `url` (required)") | No |
| VS Code | Yes (`"type": "http"`) | No |
| JetBrains AI Assistant | Yes (Streamable HTTP; SSE for legacy) | No, but UI only |
| Junie | Yes ("Remote: connect to a hosted server via HTTP/HTTPS") | No |

---

## Claude Code

Sources:
- https://code.claude.com/docs/en/mcp
- https://code.claude.com/docs/en/skills
- https://code.claude.com/docs/en/memory
- https://docs.claude.com/en/docs/claude-code/hooks

Verified 2026-10-05.

**MCP (user scope).** User-scoped servers are "stored in `~/.claude.json`", at the top-level `mcpServers` key. Local-scope servers sit under that project's path inside the same file. The scope table is Local → `~/.claude.json`, Project → `.mcp.json`, User → `~/.claude.json`.

The documented CLI, which avoids hand-editing a large, concurrently written state file:

```bash
claude mcp add --transport http --scope user ast-context-cache http://127.0.0.1:7821/mcp
# or
claude mcp add-json --scope user ast-context-cache '{"type":"http","url":"http://127.0.0.1:7821/mcp"}'
```

The installer does not shell out to `claude mcp add`, because the CLI cannot produce the preview diff; it patches the same key with a comment-preserving JSON editor and re-checks the file before writing.

Resulting JSON. This shape matches the user's existing `~/.claude.json`.

```json
{
  "mcpServers": {
    "ast-context-cache": { "type": "http", "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

- **A `url` entry with no `type` is an error.** Claude Code treats a typeless entry as stdio and skips it. `"streamable-http"` is accepted as an alias for `"http"`.
- Optional fields: `headers`, `headersHelper`, `timeout`, `alwaysLoad`.
- Interactive sessions refetch tools, prompts and resources on `list_changed`. `-p` and Agent SDK sessions refresh only the tool list.
- The v2 runtime (TS SDK 2.0) uses MCP **2026-07-28** with HTTP servers that support it, and receives `list_changed` over a held-open `subscriptions/listen` stream. See docs/spikes/mcp-protocol-versions.md.
- Windows: the docs use `~`. The `%USERPROFILE%\.claude.json` mapping is **UNVERIFIED**. `CLAUDE_CONFIG_DIR` can relocate the `.claude` config home.

**Skills.** The personal location is `~/.claude/skills/<skill-name>/SKILL.md`. It loads in all projects on this machine, but not in Cowork or cloud sessions. Frontmatter takes `description`, and optionally `name`.

**Global instructions.**
- `~/.claude/CLAUDE.md` holds user instructions.
- `~/.claude/rules/*.md` holds user-level rules, which load before project rules.
- Managed policy locations, which the installer must not touch:
  - macOS `/Library/Application Support/ClaudeCode/CLAUDE.md`
  - Linux `/etc/claude-code/CLAUDE.md`
  - Windows `C:\Program Files\ClaudeCode\CLAUDE.md`

**Hooks.** User hooks go in `~/.claude/settings.json` under `"hooks"`, applying to all projects. Entries from different settings levels merge. The user's file already has `PostToolUse`, `PreCompact`, `SessionStart` and `Stop`, so the installer must append to the event arrays and not replace them.

---

## Cursor

Sources:
- https://cursor.com/docs/context/mcp
- https://cursor.com/docs/context/rules
- https://cursor.com/docs/skills
- https://cursor.com/docs/hooks

Verified 2026-10-05.

**MCP.** Global config is `~/.cursor/mcp.json` ("for tools available everywhere"); the project equivalent is `.cursor/mcp.json`. The documented transports are stdio, SSE and Streamable HTTP. The remote entry uses `url` and optional `headers`. There is no `type` for remote entries in the docs example. `type: "stdio"` is documented for stdio entries.

```json
{
  "mcpServers": {
    "ast-context-cache": { "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

The user's existing entry is `{"url":"http://localhost:7821/mcp"}`. Interpolation `${env:NAME}` and `${userHome}` is supported in `url` and `headers`. Cursor also offers `vscode.cursor.mcp.registerServer()` for extensions.

**Skills.** Global locations are `~/.cursor/skills/` and `~/.agents/skills/`. For compatibility Cursor also loads `~/.claude/skills/` and `~/.codex/skills/`. Each skill is `<name>/SKILL.md`. **Dedup risk:** if the installer writes the same skill to both `~/.claude/skills` and `~/.cursor/skills`, Cursor will see it twice. The installer therefore writes Cursor's skills to `~/.agents/skills` only when they are not already in `~/.claude/skills` or `~/.agents/skills`, and otherwise reports `covered`.

**Rules.**
- Project rules are `.cursor/rules/*.mdc` (`.md` files there are ignored), or `AGENTS.md`.
- **Global "User Rules" are defined in Customize → Rules (Settings UI) and are not file-based.** Team Rules come from the dashboard.
- **`~/.cursor/rules/` is not a documented global location.** The user has `~/.cursor/rules/*.mdc` files, but whether Cursor loads them globally is **UNVERIFIED**.
- **Installer behavior:** the installer still writes the always-apply rule to `~/.cursor/rules/ast-context-cache.mdc`, because that is where users keep global `.mdc` rules today, and the preview warns: "loading rules from the global ~/.cursor/rules directory is unverified; Settings → Rules (User Rules) is the documented path". If Cursor does not pick the rule up, paste the body of [`rules/cursor/ast-context-cache.mdc`](../rules/cursor/ast-context-cache.mdc) (without the frontmatter) into Settings → Rules → User Rules.

**Hooks.** Global hooks are `~/.cursor/hooks.json` with `{"version":1,"hooks":{"<event>":[{"command":"…"}]}}`, and scripts conventionally go in `~/.cursor/hooks/`. Project hooks are `.cursor/hooks.json`. MDM system paths: macOS `/Library/Application Support/Cursor/hooks.json`, Linux `/etc/cursor/hooks.json`, Windows `C:\ProgramData\Cursor\hooks.json`.

Windows path for `~/.cursor/mcp.json`: the docs say "home directory". `%USERPROFILE%\.cursor\mcp.json` is **UNVERIFIED** but implied.

---

## OpenCode

Sources:
- https://opencode.ai/docs/mcp-servers/
- https://opencode.ai/docs/config/
- https://opencode.ai/docs/rules/
- https://opencode.ai/docs/skills/

Verified 2026-10-05.

**Config file.** Global config is `~/.config/opencode/opencode.json`. JSON and JSONC are both supported; the user actually has `~/.config/opencode/opencode.jsonc`. The installer must:
- detect `.jsonc` versus `.json`;
- preserve comments and trailing commas;
- never create a second file next to an existing one.

Configs are **merged**, not replaced, across these layers: remote `.well-known/opencode` → global → `OPENCODE_CONFIG` → project `opencode.json` → `.opencode/` → `OPENCODE_CONFIG_CONTENT` → managed. Managed config, which the installer must not touch, lives in macOS `/Library/Application Support/opencode/`, Linux `/etc/opencode/`, Windows `%ProgramData%\opencode`. The Windows user-config path is **UNVERIFIED**.

**MCP.** Key `mcp`. Remote options are `type` (must be `"remote"`), `url`, `enabled`, `headers`, `oauth` (object or `false`) and `timeout` (ms, default 5000).

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "ast-context-cache": {
      "type": "remote",
      "url": "http://127.0.0.1:7821/mcp",
      "enabled": true,
      "oauth": false
    }
  }
}
```

- `"oauth": false` is recommended for a local, no-auth server. It stops OpenCode from starting OAuth discovery after a 401. The installer writes only `type`, `url`, and `enabled`; add `oauth` or `timeout` by hand if you want them (the entry then reports `modified_by_user`).
- The user's live entry uses `type: "remote"`, `url: "http://localhost:7821/mcp"`, `enabled: true` and `timeout: 30000`.
- MCP tools are registered as `<server>_<tool>`, for example `ast-context-cache_*`.
- The key is `mcp`, not `mcpServers` (pre-4.0 repo docs had this wrong).

**Rules.**
- Global instructions are `~/.config/opencode/AGENTS.md`, falling back to `~/.claude/CLAUDE.md` if absent (`OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=1` disables the fallback).
- Extra files can be added via `"instructions": ["path or glob or URL", …]` in the global config.
- `~/.config/opencode/rules/` exists on the user's machine but is **not** a documented OpenCode location. It is probably loaded through a plugin or `instructions`, which is UNVERIFIED.

**Skills.** Global locations are `~/.config/opencode/skills/<name>/SKILL.md`, `~/.claude/skills/<name>/SKILL.md` and `~/.agents/skills/<name>/SKILL.md`. Same dedup risk as Cursor.

**Hooks.** There is no declarative hooks file. Hooks are implemented as plugins in `~/.config/opencode/plugins/` or via npm `"plugin": [...]`. Treat this as "Unsupported (plugin required)" for v4.0.

---

## Claude Desktop

Sources:
- https://modelcontextprotocol.io/docs/develop/connect-local-servers
- https://modelcontextprotocol.io/docs/develop/build-server
- https://modelcontextprotocol.io/docs/develop/connect-remote-servers

Verified 2026-10-05.

**Config path** (Settings → Developer → Edit Config):

| OS | Path |
| - | - |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` (shown in the build-server quickstart) |

**Entry shape.** Under the `mcpServers` key the documented fields are `command`, `args`, `env`, and `cwd` (the Ruby example uses `cwd`). Every documented example is a **stdio command**. **Remote servers are added through Settings → Connectors → "Add custom connector"**, which claude.ai brokers, so they cannot reach a local `127.0.0.1` server. The installer therefore needs a **stdio bridge**:

```json
{
  "mcpServers": {
    "ast-context-cache": {
      "command": "/absolute/path/to/mcp-local",
      "args": ["bridge", "http://127.0.0.1:7821/mcp"]
    }
  }
}
```

- The bridge is `mcp-local bridge <url>` from the [mcp-local](https://github.com/coma-toast/mcp-local) companion release (a stdio → Streamable HTTP bridge). The installer writes the absolute path it finds on `PATH`, because Claude Desktop does not inherit the shell `PATH`.
- Fallback when `mcp-local` is not installed: `"command":"npx","args":["-y","mcp-remote","http://127.0.0.1:7821/mcp"]`. Whether `mcp-remote` is compatible with a dual-era or 2026-07-28 server is **UNVERIFIED**, so the preview warns, and it needs Node.js.
- `{"command":"http","url":…}`, which pre-4.0 repo docs showed, is not a valid Claude Desktop entry.
- Restart requires a full quit of the app.

**Skills, instructions, hooks.** None are documented as local files for Claude Desktop. Mark all three Unsupported.

---

## Codex CLI

Sources:
- https://developers.openai.com/codex/mcp
- https://developers.openai.com/codex/guides/agents-md
- https://developers.openai.com/codex/skills
- https://developers.openai.com/codex/hooks

Verified 2026-10-05.

**MCP.** Config is `~/.codex/config.toml`, or `$CODEX_HOME/config.toml`. The project equivalent is `.codex/config.toml`, for trusted projects only. The ChatGPT desktop app, Codex CLI and IDE extension share this file.

**Streamable HTTP is natively supported** with no experimental flag. Fields:
- `url` (required)
- `bearer_token_env_var`
- `http_headers`
- `env_http_headers`
- `http_headers_helper`
- `auth`
- `enabled`, `enabled_tools`, `disabled_tools`
- `startup_timeout_sec`, `tool_timeout_sec`

CLI: `codex mcp add ast-context-cache --url http://127.0.0.1:7821/mcp`

```toml
[mcp_servers.ast-context-cache]
url = "http://127.0.0.1:7821/mcp"
enabled = true
startup_timeout_sec = 20
```

- A TOML bare key may contain `-`.
- The installer must preserve the user's other tables (`[desktop]`, `[plugins.*]`, `[mcp_servers.node_repl]`, …). Use a TOML round-trip editor, or append only the `[mcp_servers.ast-context-cache]` table.
- Codex "reads the MCP `instructions` field returned during initialization", so it uses the legacy handshake path. Keep the first 512 chars of `instructions` self-contained.

**Global instructions.** `~/.codex/AGENTS.md`, or `AGENTS.override.md` which takes precedence. Codex uses the first non-empty file in the Codex home, and the combined limit is `project_doc_max_bytes`, 32 KiB by default. The user currently has **no** `~/.codex/AGENTS.md`.

**Skills.** The user scope is `$HOME/.agents/skills/<name>/SKILL.md`. Admin scope is `/etc/codex/skills`. Repo scope is `.agents/skills`. Symlinks are followed. `~/.codex/skills` is **not** listed by Codex's own docs, although Cursor reads it.

**Hooks.** `~/.codex/hooks.json`, or inline `[hooks]` / `[[hooks.<Event>]]` tables in `~/.codex/config.toml`. The repo equivalents are `.codex/hooks.json` and `.codex/config.toml`. All layers load. Use one representation per layer, because mixing both makes Codex warn. Handler types are `command` and `mcp_tool`.

---

## VS Code (GitHub Copilot agent mode)

Sources:
- https://code.visualstudio.com/docs/copilot/customization/mcp-servers
- https://code.visualstudio.com/docs/copilot/customization/custom-instructions

Verified 2026-10-05.

**MCP locations documented:**
- Workspace `.vscode/mcp.json` uses a `servers` object.
- Workspace `.mcp.json` uses `mcpServers` (the portable format).
- **User profile `mcp.json`**, opened with "MCP: Open User Configuration", uses a `servers` object. Each VS Code profile has its own.
- **User, portable:** `$COPILOT_HOME/mcp-config.json`, or `~/.copilot/mcp-config.json`, with `mcpServers`.
- The "MCP: Add Server" flow labels `.vscode/mcp.json` and the user-profile destination as **deprecated** and says to prefer the portable destinations. The Agent Host reads `~/.copilot/mcp-config.json` natively.

User-profile entry (documented shape):

```json
{
  "servers": {
    "ast-context-cache": { "type": "http", "url": "http://127.0.0.1:7821/mcp" }
  }
}
```

User-profile path, default profile:

| OS | Path | Status |
| - | - | - |
| macOS | `~/Library/Application Support/Code/User/mcp.json` | Observed on this machine |
| Linux | `~/.config/Code/User/mcp.json` | **UNVERIFIED**, inferred from the standard VS Code user-data dir |
| Windows | `%APPDATA%\Code\User\mcp.json` | **UNVERIFIED**, same inference |

Non-default profiles live under `User/profiles/<id>/`, which is UNVERIFIED for `mcp.json`.

Portable entry for `~/.copilot/mcp-config.json`. The `mcpServers` key is documented. The inner `{"type":"http","url":…}` shape is **UNVERIFIED**.

```json
{ "mcpServers": { "ast-context-cache": { "type": "http", "url": "http://127.0.0.1:7821/mcp" } } }
```

**Warning:** the user's current VS Code `mcp.json` contains a stray `,,`, which is invalid JSON. VS Code tolerates it, but a strict JSON parser will fail. The installer must use a tolerant JSONC parser, or refuse to edit and print manual steps.

**Instructions.**
- Agent Host user scope: `~/.copilot/copilot-instructions.md` (always-on) and `~/.copilot/instructions/**/*.instructions.md`.
- The Claude harness user rules are in `~/.claude/rules`.
- The Local agent keeps user instructions in VS Code profile storage and also reads `~/.claude/CLAUDE.md` when `chat.useClaudeMdFile` is enabled.

**Skills and hooks.** Both are supported features. Their user-scope paths were not fetched, so they are **UNVERIFIED**.

---

## JetBrains: AI Assistant and Junie

Sources:
- https://www.jetbrains.com/help/ai-assistant/mcp.html
- https://junie.jetbrains.com/docs/junie-cli-mcp-configuration.html

Verified 2026-10-05.

**AI Assistant.** **Unsupported (manual steps)** — there is no documented global file. The path is Settings → Tools → AI Assistant → Model Context Protocol (MCP) → Add. Choose transport, paste the JSON, set "Server level" to Global, then OK and Apply. An "Import from Claude" button imports Claude Desktop config. The transports are STDIO and Streamable HTTP, plus SSE for legacy. Remote JSON:

```json
{ "mcpServers": { "ast-context-cache": { "url": "http://127.0.0.1:7821/mcp" } } }
```

Organizations may preconfigure or lock MCP via JetBrains IDE Services or Central. The `.idea/mcp.json` path that pre-4.0 repo docs showed is not documented.

**Junie (CLI and IDE plugin).**
- User scope is **`~/.junie/mcp/mcp.json`**, and project scope is `.junie/mcp/mcp.json`. The IDE's Tools → Junie → MCP Settings save to `~/.junie/mcp/mcp.json`.
- Remote HTTP and HTTPS servers are supported.
- CLI flags `--mcp-default-locations` and `--mcp-location <path>` change where Junie looks.
- The JSON structure block did not render (JS-collapsed). The remote entry shape `{"mcpServers":{"<name>":{"url":…}}}` is **UNVERIFIED** and assumed to match AI Assistant.
- Junie documents "Agent skills", "Guidelines and memory" and "Hooks" pages. Their paths are **UNVERIFIED**.

---

## Origin / Accept behaviour

The policy below is what `internal/httpguard` implements for `/mcp`, the dashboard's mutating routes, and its WebSocket: a missing Origin is allowed, a loopback Origin on any port is allowed, any other Origin gets 403 (`forbidden origin`), and a non-loopback `Host` gets 403 (`forbidden host`) unless the server listens on a wildcard address such as Docker's `0.0.0.0`.

Research notes (2026-10-05):

The spec says servers MUST validate `Origin`, and since 2025-11-25 a present-but-invalid Origin MUST get 403. **None of the hosts above document whether they send an `Origin` header (UNVERIFIED for all).** Native desktop and CLI clients are non-browser HTTP clients and usually omit it. Browser-based tools such as the MCP Inspector web UI or a dashboard do send it.

Recommended policy (as researched):

1. **Origin absent** → allow. Native clients rely on this, and browsers always send Origin on cross-origin POST, so DNS-rebinding requests are still caught.
2. **Origin present** with host `localhost`, `127.0.0.1` or `[::1]` (any port, which covers the dashboard on :7830) → allow.
3. **Origin present with any other host** → `403`, with a JSON-RPC error body without `id`.
4. Also check that `Host` is `127.0.0.1:7821`, `localhost:7821` or `[::1]:7821` as defence in depth against DNS rebinding.
5. **Accept:** be lenient.
   - If `Accept` includes `text/event-stream`, SSE may be used. Otherwise always reply `application/json`.
   - Do not reject a missing or partial `Accept` with 406, even though the spec says clients MUST send both types. Clients vary.
   - GET without `text/event-stream` → 405.

## UNVERIFIED items (rollup)

- Windows user paths for Claude Code (`~/.claude.json`), Cursor, OpenCode, VS Code (`%APPDATA%\Code\User\mcp.json`) and VS Code on Linux.
- Whether Cursor loads `~/.cursor/rules/*.mdc` globally. Docs say global rules are UI-only User Rules.
- Whether OpenCode loads `~/.config/opencode/rules/`. This directory is not documented.
- VS Code: the inner entry shape in `~/.copilot/mcp-config.json`, and the user-scope skills and hooks paths.
- Junie: the remote entry JSON shape, and the guidelines, skills and hooks paths.
- JetBrains AI Assistant: any on-disk global config file. It is documented as UI-only.
- Claude Desktop bridge: whether `mcp-remote` is compatible with a 2026-07-28 or dual-era server (the `mcp-local bridge` path is the supported one).
- Origin header behaviour of every client.
- OpenCode's HTTP sub-transport, Streamable versus SSE. The docs only say `remote`.
