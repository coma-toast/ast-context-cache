package installer

import (
	"os"

	"github.com/coma-toast/ast-context-cache/instructions"
)

// openCodeSpec: MCP under the "mcp" key of ~/.config/opencode/opencode.jsonc (or .json) as a
// remote server, the instruction block in ~/.config/opencode/AGENTS.md, and the shared skills
// rule. OpenCode hooks need a plugin, so they are unsupported.
func (s *realService) openCodeSpec() targetSpec {
	cfg := s.homePath(".config", "opencode", "opencode.jsonc")
	if _, err := os.Stat(cfg); err != nil {
		cfg = s.homePath(".config", "opencode", "opencode.json")
	}
	return targetSpec{id: TargetOpenCode, name: "OpenCode", units: map[Component]unit{
		ComponentMCP: jsonEntryUnit{
			file:  cfg,
			block: "mcp",
			entry: jsonObj{{"type", "remote"}, {"url", s.cfg.MCPURL}, {"enabled", true}},
		},
		ComponentSkills: s.agentsSkills("OpenCode"),
		ComponentRules:  mdBlockUnit{file: s.homePath(".config", "opencode", "AGENTS.md"), body: instructions.AgentsBlock},
		ComponentHooks:  unsupportedUnit{reason: "OpenCode hooks require a plugin; handoff hooks are Claude Code only"},
	}}
}
