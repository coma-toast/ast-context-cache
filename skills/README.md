# Agent skills (portable)

Copy-paste and editor-agnostic instruction blocks live here.

| Path | Audience |
|------|----------|
| [agents/SKILL.md](agents/SKILL.md) | MCP JSON for OpenCode, Cursor, Claude, VS Code, JetBrains + pasteable AGENTS block |
| [install/SKILL.md](install/SKILL.md) | Install and troubleshoot ast-mcp |
| [usage/SKILL.md](usage/SKILL.md) | MCP tool selection, RAG, token tips |
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

Optional later: `make sync-cursor-skills` to automate the copy.
