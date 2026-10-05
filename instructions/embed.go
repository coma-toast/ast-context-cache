// Package instructions embeds the canonical instruction block the installer places, between
// version-stamped markers, in host instruction files such as ~/.claude/CLAUDE.md and AGENTS.md (IN-11).
package instructions

import _ "embed"

// AgentsBlock is the shared CLAUDE.md / AGENTS.md block body.
//
//go:embed agents-block.md
var AgentsBlock string
