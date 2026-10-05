package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
)

// setupHandoffToolTest indexes the trail fixture project, pins every handoff flag on (callers
// override with setFlagEnvs), and installs a running handoff service as the Default.
func setupHandoffToolTest(t *testing.T) string {
	t.Helper()
	t.Setenv("AST_MCP_TIER", "complete")
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	t.Cleanup(func() { SetConfig(origCfg) })
	project, _ := indexedPython(t, "llamacpp.py", trailClientPy)
	t.Cleanup(db.FlushWriteBuffers)
	setFlagEnvs(t, map[string]string{
		"AST_FEATURE_HANDOFF": "true", "AST_FEATURE_HANDOFF_SCRATCHPAD": "true",
		"AST_FEATURE_HANDOFF_CLAIMS": "true", "AST_FEATURE_HANDOFF_LIVE_TRAIL": "true",
	})
	ctx, cancel := context.WithCancel(context.Background())
	prev := handoff.Default()
	handoff.Start(ctx, nil)
	t.Cleanup(func() {
		cancel()
		handoff.SetDefault(prev)
	})
	return project
}

// callToolTexts runs a tools/call and returns every content item's text plus isError.
func callToolTexts(t *testing.T, name string, arguments map[string]any) ([]string, bool) {
	t.Helper()
	req := JSONRPCRequest{JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: map[string]any{"name": name, "arguments": arguments}}
	rec := httptest.NewRecorder()
	handleToolCall(rec, req)
	var resp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp), rec.Body.String())
	var texts []string
	for _, c := range resp.Result.Content {
		texts = append(texts, c.Text)
	}
	return texts, resp.Result.IsError
}

// mustCall runs a tool call that must succeed.
func mustCall(t *testing.T, name string, arguments map[string]any) map[string]any {
	t.Helper()
	out, isErr := callTool(t, name, arguments)
	require.False(t, isErr, "%s %v: %v", name, arguments, out)
	return out
}

// openChild creates a handoff from parent and opens one child, returning the ref and child id.
func openChild(t *testing.T, project, parent string) (ref, child string) {
	t.Helper()
	created := mustCall(t, toolHandoff, map[string]any{"action": "create", "session_id": parent, "project_path": project, "brief": "Investigate load_model"})
	ref, _ = created["handoff"].(string)
	opened := mustCall(t, toolOpenHandoff, map[string]any{"handoff": ref, "project_path": project})
	child, _ = opened["session_id"].(string)
	require.NotEmpty(t, child)
	return ref, child
}

func capsule(t *testing.T, project, sid string) map[string]any {
	t.Helper()
	return mustCall(t, "get_context_capsule", map[string]any{"query": "load_model", "project_path": project, "session_id": sid})
}

func anyExplored(results any) bool {
	for _, r := range resultMaps(results) {
		if explored, _ := r["parent_explored"].(bool); explored {
			return true
		}
	}
	return false
}

func TestHandoffToolsRegisteredAtCore(t *testing.T) {
	byName := map[string]Tool{}
	var order []string
	for _, tool := range GetTools() {
		byName[tool.Name] = tool
		order = append(order, tool.Name)
	}
	for _, name := range []string{toolHandoff, toolOpenHandoff, toolScratchpad} {
		tool, ok := byName[name]
		require.True(t, ok, "missing tool %s", name)
		assert.Equal(t, TierCore, tool.Tier, name)
	}
	at := strings.Index(strings.Join(order, ","), "recall_memory,handoff,open_handoff,scratchpad,")
	assert.GreaterOrEqual(t, at, 0, "the handoff tools follow recall_memory: %v", order)
	assert.True(t, strings.HasPrefix(byName[toolOpenHandoff].Description, "If your prompt contains [handoff hof_…], call open_handoff before any search."))
	assert.Contains(t, byName[toolScratchpad].Description, "advisory")
}

// TS-3: the three tools add at most 1,200 tokens to tools/list.
func TestHandoffToolSchemasWithinBudget(t *testing.T) {
	data, err := json.Marshal([]Tool{handoffTool(), openHandoffTool(), scratchpadTool()})
	require.NoError(t, err)
	tokens := db.EstimateTokens(string(data))
	t.Logf("handoff tool schemas: %d tokens", tokens)
	assert.LessOrEqual(t, tokens, 1200)
}

func TestHandoffToolsHiddenWhenFeatureOff(t *testing.T) {
	setupHandoffToolTest(t)
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": "false"})
	for _, tool := range FilterTools(ServerConfig{ActiveTier: TierComplete, CodeMode: true}) {
		assert.NotContains(t, []string{toolHandoff, toolOpenHandoff, toolScratchpad}, tool.Name)
	}
	for _, name := range []string{toolHandoff, toolOpenHandoff, toolScratchpad} {
		texts, isErr := callToolTexts(t, name, map[string]any{"action": "list", "session_id": "s"})
		assert.True(t, isErr, name)
		require.Len(t, texts, 1)
		assert.Contains(t, texts[0], "feature_disabled", name)
	}
}

// AC30 / FF-7: turning feature_handoff off and on again deletes nothing; trees made before are
// reachable afterwards.
func TestHandoffDataSurvivesFlagToggle(t *testing.T) {
	project := setupHandoffToolTest(t)
	ref, child := openChild(t, project, "parent-toggle")
	mustCall(t, toolScratchpad, map[string]any{"action": "post", "session_id": child, "type": "finding", "text": "load_model reads the config"})
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": "false"})
	texts, isErr := callToolTexts(t, toolOpenHandoff, map[string]any{"action": "resume", "handoff": ref, "session_id": child})
	require.True(t, isErr)
	assert.Contains(t, texts[0], "feature_disabled")
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF": "true"})
	resumed := mustCall(t, toolOpenHandoff, map[string]any{"action": "resume", "handoff": ref, "session_id": child})
	assert.Equal(t, child, resumed["session_id"])
	assert.Contains(t, resumed["brief"], "Investigate load_model")
	read := mustCall(t, toolScratchpad, map[string]any{"action": "read", "session_id": child, "include_own": true})
	assert.Contains(t, fmt.Sprint(read["entries"]), "load_model reads the config")
}

func TestScratchpadClaimsFlagGatesActions(t *testing.T) {
	project := setupHandoffToolTest(t)
	_, child := openChild(t, project, t.Name())
	setFlagEnvs(t, map[string]string{"AST_FEATURE_HANDOFF_CLAIMS": "false"})
	out, isErr := callTool(t, toolScratchpad, map[string]any{"action": "claim", "session_id": child, "key": "llamacpp.py"})
	assert.True(t, isErr)
	assert.Equal(t, "feature_disabled", out["error"])
	details, _ := out["details"].(map[string]any)
	assert.Equal(t, flags.KeyHandoffClaims, details["flag"])
	assert.NotEmpty(t, out["suggestions"])
	out = mustCall(t, toolScratchpad, map[string]any{"action": "post", "session_id": child, "type": "finding", "text": "load_model returns its argument"})
	assert.Positive(t, out["id"])
}

func TestHandoffToolErrors(t *testing.T) {
	setupHandoffToolTest(t)
	tests := []struct {
		name string
		tool string
		args map[string]any
		code string
	}{
		{name: "unknown ref", tool: toolOpenHandoff, args: map[string]any{"handoff": "hof_0000000000000000"}, code: string(handoff.CodeHandoffNotFound)},
		{name: "malformed ref", tool: toolOpenHandoff, args: map[string]any{"handoff": "nope"}, code: "invalid_input"},
		{name: "resume without session", tool: toolOpenHandoff, args: map[string]any{"action": "resume", "handoff": "hof_0000000000000000"}, code: "invalid_input"},
		{name: "unknown action", tool: toolHandoff, args: map[string]any{"action": "explode", "session_id": "s"}, code: "invalid_input"},
		{name: "create without brief", tool: toolHandoff, args: map[string]any{"action": "create", "session_id": "s"}, code: "invalid_input"},
		{name: "bad expand items", tool: toolOpenHandoff, args: map[string]any{"action": "expand", "handoff": "hof_0000000000000000", "session_id": "s", "items": []any{"x"}}, code: "invalid_input"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, isErr := callTool(t, tt.tool, tt.args)
			assert.True(t, isErr)
			assert.Equal(t, tt.code, out["error"], "%v", out)
			assert.Contains(t, out, "message")
		})
	}
}

func TestHandoffToolsWithoutService(t *testing.T) {
	setupHandoffToolTest(t)
	handoff.SetDefault(nil)
	out, isErr := callTool(t, toolHandoff, map[string]any{"action": "list", "session_id": "s"})
	assert.True(t, isErr)
	assert.Equal(t, "handoff service not started", out["error"])
}

// End to end: a parent searches and hands off, two children open it, search, share the
// scratchpad, queue on a claim, and complete; the parent collects.
func TestHandoffToolsEndToEnd(t *testing.T) {
	project := setupHandoffToolTest(t)
	parent := t.Name()
	out := capsule(t, project, parent)
	require.NotEmpty(t, out["results"])
	assert.NotContains(t, out, "handoff", "no annotations outside a tree")

	created := mustCall(t, toolHandoff, map[string]any{"action": "create", "session_id": parent, "project_path": project, "brief": "Investigate load_model", "label": "loader"})
	ref, _ := created["handoff"].(string)
	assert.Regexp(t, `^hof_[0-9a-f]{16}$`, ref)
	assert.Contains(t, created["stub"], "[handoff "+ref)

	opened := mustCall(t, toolOpenHandoff, map[string]any{"handoff": ref, "project_path": project})
	child1, _ := opened["session_id"].(string)
	assert.Equal(t, ref+".c1", child1)
	assert.Equal(t, "Investigate load_model", opened["brief"])
	assert.NotEmpty(t, opened["trail"], "the digest lists the parent's searches")

	// OP-9, OP-10: the child repeats the parent's search.
	out = capsule(t, project, child1)
	ann, _ := out["handoff"].(map[string]any)
	require.NotNil(t, ann, "%v", out)
	assert.Contains(t, ann, "parent_trail_match")
	assert.True(t, anyExplored(out["results"]), "a fresh child sees parent_explored: %v", out["results"])

	// SP-6: a sibling's same search matches child1's live trail. Each sibling's first look at
	// load_model is still marked, since dedup is per child.
	opened2 := mustCall(t, toolOpenHandoff, map[string]any{"handoff": ref, "project_path": project})
	child2, _ := opened2["session_id"].(string)
	assert.Equal(t, ref+".c2", child2)
	fc := mustCall(t, "get_file_context", map[string]any{"file": project + "/llamacpp.py", "project_path": project, "session_id": child2})
	assert.True(t, anyExplored(fc["symbols"]), "%v", fc)
	opened3 := mustCall(t, toolOpenHandoff, map[string]any{"handoff": ref, "project_path": project})
	child3, _ := opened3["session_id"].(string)
	rt := mustCall(t, "retrieve", map[string]any{"query": "load_model", "project_path": project, "session_id": child3, "include_docs": false})
	assert.True(t, anyExplored(rt["chunks"]), "%v", rt)
	out = capsule(t, project, child2)
	ann, _ = out["handoff"].(map[string]any)
	require.NotNil(t, ann)
	sibling, _ := ann["sibling_trail_match"].(map[string]any)
	require.NotNil(t, sibling, "%v", ann)
	assert.Equal(t, child1, sibling["author"])

	// Scratchpad between siblings.
	mustCall(t, toolScratchpad, map[string]any{"action": "post", "session_id": child1, "type": "finding", "text": "load_model returns its argument", "refs": []any{"llamacpp.py"}})
	read := mustCall(t, toolScratchpad, map[string]any{"action": "read", "session_id": child2, "types": []any{"finding"}})
	entries := resultMaps(read["entries"])
	require.Len(t, entries, 1)
	assert.Equal(t, child1, entries[0]["author"])

	// CL-4, CL-5: child2 queues behind child1 and learns of its grant on its next call.
	claim := mustCall(t, toolScratchpad, map[string]any{"action": "claim", "session_id": child1, "key": "llamacpp.py"})
	assert.Equal(t, "granted", claim["outcome"])
	claim = mustCall(t, toolScratchpad, map[string]any{"action": "claim", "session_id": child2, "key": "llamacpp.py", "reason": "edit"})
	assert.Equal(t, "queued", claim["outcome"])
	release := mustCall(t, toolScratchpad, map[string]any{"action": "release", "session_id": child1, "key": "llamacpp.py"})
	assert.Equal(t, child2, release["granted_to"])
	texts, isErr := callToolTexts(t, "index_status", map[string]any{"project_path": project, "session_id": child1})
	require.False(t, isErr)
	assert.Len(t, texts, 1, "the releaser gets no notice")
	texts, isErr = callToolTexts(t, toolScratchpad, map[string]any{"action": "read", "session_id": child2})
	require.False(t, isErr)
	require.Len(t, texts, 2)
	assert.Equal(t, "[claims_granted] llamacpp.py (scratchpad)", texts[1])
	texts, _ = callToolTexts(t, toolScratchpad, map[string]any{"action": "read", "session_id": child2})
	assert.Len(t, texts, 1, "a grant is announced once")

	// RT: child1 completes with a summary stub.
	content := "FACT: load_model returns name\n" + strings.Repeat("Detailed notes about the loader. ", 40)
	done := mustCall(t, toolHandoff, map[string]any{"action": "complete", "session_id": child1, "content": content, "summary": "load_model is a passthrough", "changed_files": []any{"llamacpp.py"}})
	assert.Equal(t, "done", done["status"])
	assert.Equal(t, "load_model is a passthrough", done["summary"])
	assert.Contains(t, done["stub"], ref)

	collected := mustCall(t, toolHandoff, map[string]any{"action": "collect", "session_id": parent})
	statuses := map[string]string{}
	for _, c := range resultMaps(collected["children"]) {
		sid, _ := c["session_id"].(string)
		statuses[sid], _ = c["status"].(string)
	}
	assert.Equal(t, map[string]string{child1: "done", child2: "open", child3: "open"}, statuses)
	listed := mustCall(t, toolHandoff, map[string]any{"action": "list", "session_id": parent})
	assert.Len(t, listed["handoffs"], 1)
	status := mustCall(t, toolHandoff, map[string]any{"action": "status", "handoff": ref})
	assert.Equal(t, parent, status["root_session_id"])
}

// OB-2, OB-3: open and expand log their change to the child's available−delivered, so the
// rows sum to it; complete logs the result tokens the parent didn't read.
func TestHandoffToolsLogSavings(t *testing.T) {
	project := setupHandoffToolTest(t)
	parent := t.Name()
	capsule(t, project, parent)
	ref, child := openChild(t, project, parent)
	mustCall(t, toolOpenHandoff, map[string]any{"action": "expand", "handoff": ref, "session_id": child, "section": "trail", "items": "all"})
	mustCall(t, toolOpenHandoff, map[string]any{"action": "resume", "handoff": ref, "session_id": child})
	var want int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT tokens_available - tokens_delivered FROM handoff_children WHERE child_session_id = ?`, child).Scan(&want))
	content := strings.Repeat("A long result paragraph. ", 80)
	done := mustCall(t, toolHandoff, map[string]any{"action": "complete", "session_id": child, "content": content, "summary": "short"})
	db.FlushWriteBuffers()
	sumSaved := func(tool string) int {
		var n int
		require.NoError(t, db.DB.QueryRow(`SELECT COALESCE(SUM(tokens_saved), 0) FROM queries WHERE tool_name = ?`, tool).Scan(&n))
		return n
	}
	assert.Equal(t, want, sumSaved(toolOpenHandoff))
	saved, _ := done["tokens_saved"].(float64)
	assert.Positive(t, saved)
	assert.Equal(t, int(saved), sumSaved(toolHandoff), "create logs no savings; complete logs OB-3")
}

func TestClaimsGrantedNoticeTruncates(t *testing.T) {
	keys := []string{"internal/handoff/claims.go", "internal/handoff/scratchpad.go", "internal/mcp/handoff_tools.go", "internal/mcp/server.go", "a/very/long/path/that/keeps/going/and/going/forever.go"}
	text := grantNotice(keys)
	assert.LessOrEqual(t, db.EstimateTokens(text), maxGrantNoticeTokens)
	assert.Contains(t, text, "more (scratchpad)")
	assert.True(t, strings.HasPrefix(text, "[claims_granted] internal/handoff/claims.go"))
	assert.Equal(t, "[claims_granted] x.go (scratchpad)", grantNotice([]string{"x.go"}))
	assert.Equal(t, "[claims_granted] +1 more (scratchpad)", grantNotice([]string{strings.Repeat("k", 200)}))
}
