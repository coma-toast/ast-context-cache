package installer

// vsCodeSpec: the user-profile mcp.json "servers" map with an http entry. VS Code tolerates
// files a JSONC parser rejects; then the target aborts with manual steps and the file is never
// rewritten. Instructions, skills, and hooks paths are unverified, so they are unsupported.
func (s *realService) vsCodeSpec() targetSpec {
	file := s.homePath(".config", "Code", "User", "mcp.json")
	if s.cfg.GOOS == "darwin" {
		file = s.homePath("Library", "Application Support", "Code", "User", "mcp.json")
	}
	const unverified = "VS Code's user-scope location for this component is unverified"
	return targetSpec{id: TargetVSCode, name: "VS Code", units: map[Component]unit{
		ComponentMCP: jsonEntryUnit{
			file:   file,
			block:  "servers",
			entry:  jsonObj{{"type", "http"}, {"url", s.cfg.MCPURL}},
			manual: `Fix the file, or run "MCP: Open User Configuration" in VS Code and add "ast-context-cache": {"type": "http", "url": "` + s.cfg.MCPURL + `"} under "servers"`,
		},
		ComponentSkills: unsupportedUnit{reason: unverified},
		ComponentRules:  unsupportedUnit{reason: unverified},
		ComponentHooks:  unsupportedUnit{reason: "handoff hooks are Claude Code only"},
	}}
}
