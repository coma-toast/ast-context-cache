package installer

import (
	"os"

	"github.com/coma-toast/ast-context-cache/rules"
)

// openCodeSpec: MCP under the "mcp" key of ~/.config/opencode/opencode.jsonc (or .json) as a
// remote server, the instruction block in ~/.config/opencode/AGENTS.md, the shared skills, and
// the search-first nudge plugin in ~/.config/opencode/plugins.
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
		ComponentRules:  mdBlockUnit{file: s.homePath(".config", "opencode", "AGENTS.md"), body: rules.Protocol},
		ComponentHooks:  s.openCodeNudgeUnit(),
	}}
}

// openCodeNudgeUnit is the search-first nudge as a global OpenCode plugin. Handoff hooks are
// Claude Code only.
func (s *realService) openCodeNudgeUnit() unit {
	content, err := openCodeNudgePlugin(s.cfg.MCPURL)
	if err != nil {
		return unsupportedUnit{reason: err.Error()}
	}
	return nudgePluginUnit{file: s.homePath(".config", "opencode", "plugins", openCodeNudgeFile), content: content}
}
