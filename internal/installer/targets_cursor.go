package installer

import (
	"github.com/coma-toast/ast-context-cache/rules"
)

const cursorRulesWarning = "Cursor: loading rules from the global ~/.cursor/rules directory is unverified; Settings → Rules (User Rules) is the documented path"

// cursorSpec: MCP in ~/.cursor/mcp.json (url only), the always-apply rule as
// ~/.cursor/rules/ast-context-cache.mdc, and skills in ~/.agents/skills unless Cursor already
// loads ours from ~/.claude/skills or ~/.agents/skills. Cursor hooks are not part of v4.
func (s *realService) cursorSpec() targetSpec {
	front, body := splitFrontmatter(rules.CursorRule)
	return targetSpec{id: TargetCursor, name: "Cursor", units: map[Component]unit{
		ComponentMCP: jsonEntryUnit{
			file:  s.homePath(".cursor", "mcp.json"),
			block: "mcpServers",
			entry: jsonObj{{"url", s.cfg.MCPURL}},
		},
		ComponentSkills: s.agentsSkills("Cursor"),
		ComponentRules: mdBlockUnit{
			file:        s.homePath(".cursor", "rules", "ast-context-cache.mdc"),
			body:        body,
			frontmatter: front,
			warnings:    []string{cursorRulesWarning},
		},
		ComponentHooks: unsupportedUnit{reason: "handoff hooks are Claude Code only"},
	}}
}

// agentsSkills installs into ~/.agents/skills for hosts that also read ~/.claude/skills.
func (s *realService) agentsSkills(host string) skillsUnit {
	return skillsUnit{
		root:       s.homePath(".agents", "skills"),
		coverRoots: []string{s.homePath(".claude", "skills"), s.homePath(".agents", "skills")},
		host:       host,
	}
}
