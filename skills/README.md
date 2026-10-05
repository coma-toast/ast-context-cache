# Agent skills (portable)

Copy-paste and editor-agnostic instruction blocks live here. These files are also the **canonical source** the installer ships: `skills/embed.go` embeds them, and `ast-mcp install` writes each one to `<host skills dir>/ast-context-cache-<name>/SKILL.md` with `name` / `description` frontmatter added from the catalog in `embed.go` (see [docs/host-integration.md](../docs/host-integration.md)).

| Path | Audience |
|------|----------|
| [agents/SKILL.md](agents/SKILL.md) | Installer pointer, manual MCP entries for every host, pasteable AGENTS.md / CLAUDE.md block |
| [install/SKILL.md](install/SKILL.md) | Install, `ast-mcp install` / `verify` / `uninstall`, troubleshooting |
| [usage/SKILL.md](usage/SKILL.md) | MCP tool selection, RAG, token tips, **virtual context**, **structured memory**, **subagent handoff** |
| [operator/SKILL.md](operator/SKILL.md) | Embeddings, dashboard settings, feature flags, handoff limits, log indexing (operators) |
| [tools.json.example](tools.json.example) | Per-tool tier overrides |

Other canonical installer assets: the instruction block [`instructions/agents-block.md`](../instructions/agents-block.md) (written between markers into `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.config/opencode/AGENTS.md`) and the Cursor rule [`rules/cursor/ast-context-cache.mdc`](../rules/cursor/ast-context-cache.mdc). Keep all of them consistent with [AGENTS.md](../AGENTS.md) and [docs/handoff.md](../docs/handoff.md).

## Cursor (discoverable project skills)

Cursor loads skills from [`.cursor/skills/`](../.cursor/skills/) with YAML `name` and `description` frontmatter.

**Canonical sources:** edit `skills/usage/SKILL.md`, `skills/install/SKILL.md`, `skills/operator/SKILL.md`, then re-sync the matching `.cursor/skills/*/SKILL.md`, keeping each file's frontmatter (the first four lines) unchanged. Descriptions must not contain `": "` (it breaks plain YAML scalars).

| Cursor skill | Canonical source |
|--------------|------------------|
| `.cursor/skills/ast-usage/` | `skills/usage/SKILL.md` |
| `.cursor/skills/ast-install/` | `skills/install/SKILL.md` |
| `.cursor/skills/ast-rebuild/` | maintained in-repo (repo-relative paths) |
| `.cursor/skills/ast-operator/` | `skills/operator/SKILL.md` |

**Global Cursor rule:** `~/.cursor/rules/ast-context-cache.mdc` (`alwaysApply: true`), installed by `ast-mcp install --target cursor --component rules --yes` from [`rules/cursor/ast-context-cache.mdc`](../rules/cursor/ast-context-cache.mdc).

After editing portable skills, re-sync the Cursor copies. The sync drops the canonical `# Title` line and rewrites relative links, since the copies sit one directory deeper:

```bash
cd /path/to/ast-context-cache
for pair in usage:ast-usage install:ast-install operator:ast-operator; do
  src=${pair%%:*}; dir=${pair##*:}; f=".cursor/skills/$dir/SKILL.md"
  { head -4 "$f"; echo; tail -n +3 "skills/$src/SKILL.md" | sed -E \
      -e 's#\]\(\.\./\.\./#](../../../#g' \
      -e 's#\]\(\.\./(usage|install|operator)/#](../ast-\1/#g' \
      -e 's#\]\(\.\./(agents/|tools\.json\.example)#](../../../skills/\1#g'; } > "$f.tmp" && mv "$f.tmp" "$f"
done
```

Agents should read **`AGENTS.md`** / **`CLAUDE.md`** at the repo root, or invoke the matching `.cursor/skills/` skill when the task fits its description.
