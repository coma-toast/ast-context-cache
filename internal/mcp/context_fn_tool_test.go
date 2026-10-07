package mcp

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/flags"
)

func setupFnToolTest(t *testing.T) string {
	t.Helper()
	t.Setenv("AST_MCP_TIER", "complete")
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	t.Cleanup(func() { SetConfig(origCfg) })
	setFlagEnvs(t, map[string]string{"AST_FEATURE_CONTEXT_FN": "true"})
	// No dbtest.Init here: TestMain opens the databases for the whole binary, and a
	// per-test Init reassigns the pools out from under neighbouring tests.
	t.Cleanup(db.FlushWriteBuffers)
	return t.TempDir()
}

// The whole point of shipping this behind feature_context_fn, default off, is that the
// tools are absent from tools/list and rejected on call until someone opts in.
func TestContextFnToolsHiddenWhenFlagOff(t *testing.T) {
	setupFnToolTest(t)
	setFlagEnvs(t, map[string]string{"AST_FEATURE_CONTEXT_FN": "false"})
	names := []string{"define_context_fn", "apply_context_fn", "list_context_fns"}
	for _, tool := range FilterTools(ServerConfig{ActiveTier: TierComplete, CodeMode: true}) {
		assert.NotContains(t, names, tool.Name)
	}
	for _, name := range names {
		texts, isErr := callToolTexts(t, name, map[string]any{"name": "x", "pattern": "y"})
		assert.True(t, isErr, name)
		require.Len(t, texts, 1)
		assert.Contains(t, texts[0], "feature_disabled", name)
	}
}

func TestContextFnToolsVisibleWhenFlagOn(t *testing.T) {
	setupFnToolTest(t)
	seen := map[string]bool{}
	for _, tool := range FilterTools(ServerConfig{ActiveTier: TierComplete, CodeMode: true}) {
		seen[tool.Name] = true
	}
	for _, name := range []string{"define_context_fn", "apply_context_fn", "list_context_fns"} {
		assert.True(t, seen[name], name)
	}
}

// End-to-end through the MCP boundary, which is where the phase-1 work had no coverage:
// the point of the tool is that a define followed by an apply returns real token deltas.
func TestDefineThenApplyContextFnThroughMCP(t *testing.T) {
	project := setupFnToolTest(t)
	stored, err := contextnotes.Store("sess-mcp", "dead ends: grid(0.822)\nbest: 0.9992", "plan", project, "", "", nil, nil)
	require.NoError(t, err)

	texts, isErr := callToolTexts(t, "define_context_fn", map[string]any{
		"name": "compact_turns", "pattern": `dead ends: [^\n]*`, "replacement": "",
		"description": "drop dead-end candidates", "session_id": "sess-mcp",
	})
	require.False(t, isErr, texts)
	require.Contains(t, texts[0], "compact_turns")

	texts, isErr = callToolTexts(t, "apply_context_fn", map[string]any{
		"name": "compact_turns", "refs": stored.Ref,
	})
	require.False(t, isErr, texts)
	require.Contains(t, texts[0], "tokens_reclaimed")

	fetched, err := contextnotes.Fetch([]string{stored.Ref}, "sess-mcp", "")
	require.NoError(t, err)
	require.Len(t, fetched.Notes, 1)
	assert.NotContains(t, fetched.Notes[0].Content, "dead ends")

	texts, isErr = callToolTexts(t, "list_context_fns", map[string]any{"project_path": project})
	require.False(t, isErr, texts)
	require.Contains(t, texts[0], "compact_turns")
}

// A define that cannot possibly match must fail at the tool boundary, before anything
// is stored.
func TestDefineContextFnRejectsBadPatternOverMCP(t *testing.T) {
	setupFnToolTest(t)
	texts, isErr := callToolTexts(t, "define_context_fn", map[string]any{"name": "bad", "pattern": "[z-a]"})
	require.True(t, isErr)
	require.Len(t, texts, 1)
	assert.Contains(t, texts[0], "invalid pattern")
}

func TestApplyContextFnUnknownNameOverMCP(t *testing.T) {
	setupFnToolTest(t)
	texts, isErr := callToolTexts(t, "apply_context_fn", map[string]any{"name": "missing"})
	require.True(t, isErr)
	require.Len(t, texts, 1)
	assert.True(t, strings.Contains(texts[0], "no such context function"), texts[0])
}

// The flag default must stay off in the registry itself, not just in this box's env.
func TestContextFnFlagDefaultsOff(t *testing.T) {
	st := flags.State()
	for _, s := range st {
		if s.Key != flags.KeyContextFn {
			continue
		}
		assert.False(t, s.Default, "feature_context_fn must ship default-off")
		assert.Equal(t, "AST_FEATURE_CONTEXT_FN", s.Env)
		return
	}
	t.Fatal("feature_context_fn not registered")
}
