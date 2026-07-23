# Agent skills (portable)

Copy-paste and editor-agnostic instruction blocks live here.

| Path | Audience |
|------|----------|
| [agents/SKILL.md](agents/SKILL.md) | MCP JSON for OpenCode, Cursor, Claude, VS Code, JetBrains + pasteable AGENTS block |
| [install/SKILL.md](install/SKILL.md) | Install and troubleshoot ast-mcp |
| [usage/SKILL.md](usage/SKILL.md) | MCP tool selection, RAG, token tips, **virtual context** + **structured memory** |
| [operator/SKILL.md](operator/SKILL.md) | Embeddings, dashboard settings, log indexing (operators) |
| [tools.json.example](tools.json.example) | Per-tool tier overrides |

## Cursor (discoverable project skills)

Cursor loads skills from [`.cursor/skills/`](../.cursor/skills/) with YAML `name` and `description` frontmatter.

**Canonical sources:** edit `skills/usage/SKILL.md`, `skills/install/SKILL.md`, `skills/operator/SKILL.md`, then copy the body (below the frontmatter) into the matching `.cursor/skills/*/SKILL.md`, keeping each file’s frontmatter unchanged.

| Cursor skill | Canonical source |
|--------------|------------------|
| `.cursor/skills/ast-usage/` | `skills/usage/SKILL.md` |
| `.cursor/skills/ast-install/` | `skills/install/SKILL.md` |
| `.cursor/skills/ast-rebuild/` | maintained in-repo (repo-relative paths) |
| `.cursor/skills/ast-operator/` | `skills/operator/SKILL.md` |

**Global Cursor rule:** `~/.cursor/rules/ast-context-cache.mdc` (`alwaysApply: true`) — full tool list + compaction policy for all workspaces. Source: [AGENTS.md](../AGENTS.md) or [skills/agents/SKILL.md](agents/SKILL.md).

After editing portable skills, re-sync Cursor copies:

```bash
cd /path/to/ast-context-cache
for pair in usage:ast-usage operator:ast-operator; do
  IFS=: read -r src dir <<< "$pair"
  { head -4 ".cursor/skills/$dir/SKILL.md"; echo; tail -n +2 "skills/$src/SKILL.md"; } > ".cursor/skills/$dir/SKILL.md"
done
```

Agents should read **`AGENTS.md`** / **`CLAUDE.md`** at the repo root, or invoke the matching `.cursor/skills/` skill when the task fits its description.
