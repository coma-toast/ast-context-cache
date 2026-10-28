package rules

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// The protocol is the only rule text agents on Claude Code, Codex, opencode and Cursor get, so it
// must name every tool in the workflow and stay short enough to be read.
func TestProtocolCoversWorkflow(t *testing.T) {
	t.Parallel()
	for _, tool := range []string{
		"session_id", "recall_memory", "list_context", "fetch_context", "index_status", "index_files",
		"get_project_map", "get_context_capsule", "search_semantic", "retrieve", "get_file_context",
		"check_symbol_exists", "get_impact_graph", "diff_impact", "check_deletion_safety", "search_docs",
		"fetch_doc", "code_script_hints", "execute_code", "store_context", "edit_context", "store_memory",
		"search_context", "flush_context", "handoff", "open_handoff", "scratchpad",
	} {
		assert.Contains(t, Protocol, "`"+tool, tool)
	}
	assert.LessOrEqual(t, strings.Count(Protocol, "\n"), 60)
	assert.True(t, strings.HasPrefix(CursorFrontmatter, "---\n") && strings.HasSuffix(CursorFrontmatter, "\n---\n"))
	assert.Contains(t, CursorFrontmatter, "alwaysApply: true")
}
