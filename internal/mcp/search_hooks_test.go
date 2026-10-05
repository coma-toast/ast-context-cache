package mcp

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

const trailClientPy = "class LlamaCppClient:\n    def load_model(self, name):\n        return name\n\n    def unload(self):\n        pass\n"

// setupTrailTest indexes a small Python project with feature_handoff pinned to handoff.
func setupTrailTest(t *testing.T, handoff string) (project, file string) {
	t.Helper()
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	t.Cleanup(func() { SetConfig(origCfg) })
	project, file = indexedPython(t, "llamacpp.py", trailClientPy)
	// Write buffers outlive the test's DB: flush its trail rows before dbtest closes it.
	t.Cleanup(db.FlushWriteBuffers)
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": handoff})
	return project, file
}

func trailRows(t *testing.T) int {
	t.Helper()
	db.FlushWriteBuffers()
	var n int
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM search_trail`).Scan(&n))
	return n
}

func TestCapsuleRecordsSearchTrail(t *testing.T) {
	project, _ := setupTrailTest(t, "true")
	sid := t.Name()
	args := map[string]interface{}{"query": "load_model", "project_path": project, "session_id": sid}
	out, isErr := callTool(t, "get_context_capsule", args)
	require.False(t, isErr, "%v", out)
	results, _ := out["results"].([]interface{})
	require.NotEmpty(t, results)
	got := trail.ForSession(sid, 10)
	require.Len(t, got, 1, "the entry is readable before the write buffer flushes")
	e := got[0]
	assert.Equal(t, "get_context_capsule", e.Tool)
	assert.Equal(t, sid, e.SessionID)
	assert.Equal(t, "load_model", e.QueryNorm)
	assert.Equal(t, "auto", e.Mode)
	assert.Equal(t, len(results), e.HitCount)
	assert.False(t, e.ZeroHit)
	assert.Contains(t, e.TopHits, "llamacpp.py#load_model@2")

	// A repeat delivers nothing new, but the trail still counts its candidates before dedup.
	out, _ = callTool(t, "get_context_capsule", args)
	assert.Empty(t, out["results"])
	got = trail.ForSession(sid, 10)
	require.Len(t, got, 2)
	assert.Equal(t, e.HitCount, got[0].HitCount)
	assert.Equal(t, e.MatchKey(), got[0].MatchKey())
	_, ok := trail.Lookup(sid, e.MatchKey())
	assert.True(t, ok)
}

func TestCapsuleZeroHitSearchTrail(t *testing.T) {
	project, _ := setupTrailTest(t, "true")
	sid := t.Name()
	_, isErr := callTool(t, "get_context_capsule", map[string]interface{}{"query": "qqzzxxnothinghere", "project_path": project, "session_id": sid})
	require.False(t, isErr)
	got := trail.ForSession(sid, 10)
	require.Len(t, got, 1)
	assert.Equal(t, 0, got[0].HitCount)
	assert.True(t, got[0].ZeroHit)
	assert.Empty(t, got[0].TopHits)
}

func TestFileContextAndRetrieveRecordSearchTrail(t *testing.T) {
	project, file := setupTrailTest(t, "true")
	sid := t.Name()
	_, isErr := callTool(t, "get_file_context", map[string]interface{}{"file": file, "project_path": project, "session_id": sid})
	require.False(t, isErr)
	_, isErr = callTool(t, "retrieve", map[string]interface{}{"query": "load_model", "project_path": project, "session_id": sid, "include_docs": false})
	require.False(t, isErr)
	got := trail.ForSession(sid, 10)
	require.Len(t, got, 2)
	retrieve, fileCtx := got[0], got[1]
	assert.Equal(t, "get_file_context", fileCtx.Tool)
	assert.Equal(t, "llamacpp.py", fileCtx.Query)
	assert.Equal(t, 3, fileCtx.HitCount, "the class and its two methods")
	assert.Equal(t, []string{"llamacpp.py#LlamaCppClient@1", "llamacpp.py#load_model@2", "llamacpp.py#unload@5"}, fileCtx.TopHits)
	assert.Equal(t, "retrieve", retrieve.Tool)
	assert.Positive(t, retrieve.HitCount, "candidates count even though file_context already delivered them")
	assert.False(t, retrieve.ZeroHit)
}

func TestSearchTrailSkippedWithoutSession(t *testing.T) {
	project, _ := setupTrailTest(t, "true")
	_, isErr := callTool(t, "get_context_capsule", map[string]interface{}{"query": "load_model", "project_path": project})
	require.False(t, isErr)
	assert.Zero(t, trailRows(t))
}

func TestSearchTrailSkippedWhenHandoffOff(t *testing.T) {
	project, _ := setupTrailTest(t, "false")
	sid := t.Name()
	_, isErr := callTool(t, "get_context_capsule", map[string]interface{}{"query": "load_model", "project_path": project, "session_id": sid})
	require.False(t, isErr)
	assert.Empty(t, trail.ForSession(sid, 10))
	assert.Zero(t, trailRows(t))
}
