package installer

// jetBrainsSpec: JetBrains AI Assistant keeps MCP servers in IDE settings with no documented
// global file, so every component is unsupported and the MCP reason gives the manual steps.
func (s *realService) jetBrainsSpec() targetSpec {
	manual := `JetBrains AI Assistant has no documented config file. In the IDE open Settings → Tools → AI Assistant → Model Context Protocol (MCP) → Add, paste {"mcpServers":{"ast-context-cache":{"url":"` + s.cfg.MCPURL + `"}}}, set the server level to Global, then OK and Apply`
	const none = "JetBrains AI Assistant has no documented local location for this component"
	return targetSpec{id: TargetJetBrains, name: "JetBrains", units: map[Component]unit{
		ComponentMCP:    unsupportedUnit{reason: manual},
		ComponentSkills: unsupportedUnit{reason: none},
		ComponentRules:  unsupportedUnit{reason: none},
		ComponentHooks:  unsupportedUnit{reason: "handoff hooks are Claude Code only"},
	}}
}
