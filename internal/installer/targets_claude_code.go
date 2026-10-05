package installer

import (
	"github.com/coma-toast/ast-context-cache/instructions"
)

const claudeJSONWarning = "Claude Code rewrites ~/.claude.json while running; apply with Claude Code closed"

// claudeCodeSpec: user-scope MCP in ~/.claude.json (top-level mcpServers, type http), skills in
// ~/.claude/skills, the instruction block in ~/.claude/CLAUDE.md, and opt-in handoff hooks in
// ~/.claude/settings.json.
func (s *realService) claudeCodeSpec() targetSpec {
	return targetSpec{id: TargetClaudeCode, name: "Claude Code", units: map[Component]unit{
		ComponentMCP: jsonEntryUnit{
			file:     s.homePath(".claude.json"),
			block:    "mcpServers",
			entry:    jsonObj{{"type", "http"}, {"url", s.cfg.MCPURL}},
			warnings: []string{claudeJSONWarning},
		},
		ComponentSkills: skillsUnit{root: s.homePath(".claude", "skills"), host: "Claude Code"},
		ComponentRules:  mdBlockUnit{file: s.homePath(".claude", "CLAUDE.md"), body: instructions.AgentsBlock},
		ComponentHooks:  hooksUnit{file: s.homePath(".claude", "settings.json"), exe: s.cfg.Executable},
	}}
}
