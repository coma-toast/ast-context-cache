// Package rules embeds the canonical agent usage protocol the installer places in host rule and
// instruction files (IN-11). protocol.md is the single source: the markdown block in
// ~/.claude/CLAUDE.md and AGENTS.md is its body, and the Cursor rule is CursorFrontmatter plus it.
package rules

import _ "embed"

// Protocol is the shared usage protocol body for every host.
//
//go:embed protocol.md
var Protocol string

// CursorFrontmatter is the always-apply header of the Cursor rule ast-context-cache.mdc.
const CursorFrontmatter = `---
description: ast-context-cache MCP tools for token-efficient code context, virtual context edited in place as a file, structured memory, and subagent handoffs
alwaysApply: true
---
`
