package installer

import (
	"github.com/coma-toast/ast-context-cache/instructions"
)

// codexSpec: [mcp_servers.ast-context-cache] url in ~/.codex/config.toml (streamable HTTP is
// native), the instruction block in ~/.codex/AGENTS.md, and skills in ~/.agents/skills, the
// only user skill directory Codex documents.
func (s *realService) codexSpec() targetSpec {
	return targetSpec{id: TargetCodex, name: "Codex", units: map[Component]unit{
		ComponentMCP:    tomlTableUnit{file: s.homePath(".codex", "config.toml"), parent: "mcp_servers", url: s.cfg.MCPURL},
		ComponentSkills: skillsUnit{root: s.homePath(".agents", "skills"), host: "Codex"},
		ComponentRules:  mdBlockUnit{file: s.homePath(".codex", "AGENTS.md"), body: instructions.AgentsBlock},
		ComponentHooks:  unsupportedUnit{reason: "handoff hooks are Claude Code only"},
	}}
}
