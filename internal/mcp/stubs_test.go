package mcp

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
)

// The stub grammars documented in docs/stubs.md (DR-4); a producer that drifts from them
// breaks agents that parse the stubs back.
var (
	ctxRefGrammar      = regexp.MustCompile(`^ctx_[0-9a-f]{12}$`)
	ctxStubGrammar     = regexp.MustCompile(`^\[ctx_[0-9a-f]{12}\] \S.*$`)
	handoffStubGrammar = regexp.MustCompile(`^\[handoff hof_[0-9a-f]{16}\] .* — call open_handoff first$`)
	resultStubGrammar  = regexp.MustCompile(`^\[result ctx_[0-9a-f]{12} for hof_[0-9a-f]{16}(\.c[0-9]+)*\] (done|partial|failed|abandoned) — .+$`)
	offloadStubGrammar = regexp.MustCompile(`^\[ctx_[0-9a-f]{12} [a-z_]+( .+)?, [0-9]+(\.[0-9])?k? tok — head shown, fetch_context for all\]$`)
)

func TestStubGrammarContextRef(t *testing.T) {
	setupMemoryToolTest(t)
	stored := mustCall(t, "store_context", map[string]any{"session_id": t.Name(), "content": "auth design notes", "label": "auth design"})
	ref, _ := stored["ref"].(string)
	assert.Regexp(t, ctxRefGrammar, ref)
	assert.Regexp(t, ctxStubGrammar, "["+ref+"] auth design", "the chat stub an agent keeps")
}

func TestStubGrammarHandoff(t *testing.T) {
	project := setupHandoffToolTest(t)
	created := mustCall(t, toolHandoff, map[string]any{"action": "create", "session_id": t.Name(), "project_path": project, "brief": "Investigate load_model", "label": "loader"})
	stub, _ := created["stub"].(string)
	assert.Regexp(t, handoffStubGrammar, stub)
	ref, _ := created["handoff"].(string)
	opened := mustCall(t, toolOpenHandoff, map[string]any{"handoff": ref, "project_path": project})
	child, _ := opened["session_id"].(string)
	require.NotEmpty(t, child)
	done := mustCall(t, toolHandoff, map[string]any{"action": "complete", "session_id": child, "content": "FACT: load_model returns name", "summary": "load_model is a passthrough"})
	result, _ := done["stub"].(string)
	assert.Regexp(t, resultStubGrammar, result)
}

func TestStubGrammarOffload(t *testing.T) {
	for _, tc := range []struct {
		tool, target string
		tokens       int
	}{
		{"get_context_capsule", "auth middleware", 9400},
		{"get_file_context", "internal/mcp/server.go", 2100},
		{"retrieve", "", 999},
	} {
		assert.Regexp(t, offloadStubGrammar, contextnotes.OffloadStub("ctx_0123456789ab", tc.tool, tc.target, tc.tokens), tc.tool)
	}
}
