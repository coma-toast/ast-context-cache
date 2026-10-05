// Package skills embeds the canonical agent skills (IN-11). The installer copies them into each
// host's skills directory, so these files are the single source for every installed copy.
package skills

import (
	"embed"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

//go:embed agents/SKILL.md install/SKILL.md usage/SKILL.md operator/SKILL.md
var files embed.FS

// DirPrefix names every installed skill directory: <skills root>/ast-context-cache-<name>/SKILL.md.
const DirPrefix = "ast-context-cache"

// Skill is one canonical skill.
type Skill struct {
	Name        string
	Description string
	Body        string
}

// catalog lists the shipped skills in install order. Descriptions feed the frontmatter hosts use
// to decide when to load a skill, since the canonical files carry none. They are written as plain
// YAML scalars, so they must not contain ": " or start with a YAML indicator character.
var catalog = []struct{ name, description string }{
	{"usage", "Use when searching, exploring, or analyzing code with the ast-context-cache MCP tools (token-efficient search modes, session_id dedup, virtual context, structured memory), and when delegating to a subagent or when your prompt contains [handoff hof_...] (handoff, open_handoff, scratchpad)."},
	{"agents", "Use when wiring ast-context-cache into an editor or agent host - the installer, manual MCP registration entries per host, and the shared AGENTS.md / CLAUDE.md instruction block."},
	{"install", "Use when installing, configuring, or troubleshooting the ast-mcp server, registering it with editors (ast-mcp install / verify / uninstall), tool tiers, or tools.json overrides."},
	{"operator", "Use when operating the ast-mcp server - embedding backends, dashboard settings, feature flags, handoff limits, log indexing and retention, watcher ignores, and virtual context limits."},
}

// All returns the canonical skills in install order. An error means the embed directive and the
// catalog disagree, which is a build defect.
func All() ([]Skill, error) {
	out := make([]Skill, 0, len(catalog))
	for _, c := range catalog {
		b, err := files.ReadFile(c.name + "/SKILL.md")
		if err != nil {
			return nil, errs.WrapMessage("missing embedded skill", err, "skill", c.name)
		}
		out = append(out, Skill{Name: c.name, Description: c.description, Body: string(b)})
	}
	return out, nil
}

// DirName is the installed directory name for the skill.
func (s Skill) DirName() string {
	return DirPrefix + "-" + s.Name
}

// Content is the installed SKILL.md: the canonical body with name/description frontmatter added
// when the body has none.
func (s Skill) Content() string {
	if strings.HasPrefix(s.Body, "---\n") {
		return s.Body
	}
	return "---\nname: " + s.DirName() + "\ndescription: " + s.Description + "\n---\n\n" + s.Body
}
