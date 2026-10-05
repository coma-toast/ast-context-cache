package installer

// claudeDesktopSpec: Claude Desktop only launches stdio servers, so the entry runs a bridge to
// the HTTP endpoint: mcp-local when it is on PATH, otherwise npx mcp-remote. Skills, rules, and
// hooks have no documented local location.
func (s *realService) claudeDesktopSpec() targetSpec {
	file := s.homePath(".config", "Claude", "claude_desktop_config.json")
	if s.cfg.GOOS == "darwin" {
		file = s.homePath("Library", "Application Support", "Claude", "claude_desktop_config.json")
	}
	entry, warnings := s.desktopBridge()
	const none = "Claude Desktop has no documented local location for this component"
	return targetSpec{id: TargetClaudeDesktop, name: "Claude Desktop", units: map[Component]unit{
		ComponentMCP:    jsonEntryUnit{file: file, block: "mcpServers", entry: entry, warnings: warnings},
		ComponentSkills: unsupportedUnit{reason: none},
		ComponentRules:  unsupportedUnit{reason: none},
		ComponentHooks:  unsupportedUnit{reason: "handoff hooks are Claude Code only"},
	}}
}

// desktopBridge picks the stdio launch entry. Claude Desktop doesn't inherit the shell PATH, so
// mcp-local is written as an absolute path.
func (s *realService) desktopBridge() (jsonObj, []string) {
	if p, err := s.cfg.LookPath("mcp-local"); err == nil {
		return jsonObj{{"command", p}, {"args", []string{"bridge", s.cfg.MCPURL}}}, nil
	}
	entry := jsonObj{{"command", "npx"}, {"args", []string{"-y", "mcp-remote", s.cfg.MCPURL}}}
	warnings := []string{"Claude Desktop: mcp-local is not on PATH, so the entry uses npx mcp-remote as the stdio bridge (its compatibility with this server is unverified); install mcp-local and re-run install to switch"}
	if _, err := s.cfg.LookPath("npx"); err != nil {
		warnings = append(warnings, "Claude Desktop: npx is not on PATH either; install Node.js or mcp-local, or the server will not start")
	}
	return entry, warnings
}
