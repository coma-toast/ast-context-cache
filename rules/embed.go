// Package rules embeds the canonical host rule files (IN-11).
package rules

import _ "embed"

// CursorRule is the always-apply Cursor rule installed as ast-context-cache.mdc, frontmatter included.
//
//go:embed cursor/ast-context-cache.mdc
var CursorRule string
