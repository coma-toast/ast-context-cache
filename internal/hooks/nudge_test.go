package hooks

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeHealth serves /health with search_nudge set to flag (omitted when nil) and counts hits.
func fakeHealth(t *testing.T, flag *bool) (string, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/health", r.URL.Path)
		hits.Add(1)
		body := map[string]any{"status": "healthy"}
		if flag != nil {
			body["search_nudge"] = *flag
		}
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/mcp", &hits
}

func boolPtr(b bool) *bool { return &b }

func toolPayload(t *testing.T, sid, agent, tool, transcript string, input map[string]any) []byte {
	t.Helper()
	p := map[string]any{"session_id": sid, "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": input, "tool_use_id": "toolu_1"}
	if agent != "" {
		p["agent_id"], p["agent_type"] = agent, "general-purpose"
	}
	if transcript != "" {
		p["transcript_path"] = transcript
	}
	return encode(t, p)
}

func cursorPayload(t *testing.T, conv, tool string, input map[string]any) []byte {
	t.Helper()
	return encode(t, map[string]any{"conversation_id": conv, "hook_event_name": "preToolUse", "tool_name": tool, "tool_input": input})
}

// denied reports whether a Claude Code hook response denies the call.
func denied(t *testing.T, stdout string) bool {
	t.Helper()
	if stdout == "" {
		return false
	}
	out := decodeOutput(t, stdout)
	assert.Equal(t, "PreToolUse", out.HookSpecificOutput.HookEventName)
	assert.Equal(t, "deny", out.HookSpecificOutput.PermissionDecision)
	assert.Contains(t, out.HookSpecificOutput.PermissionDecisionReason, "ast-context-cache")
	return true
}

func cursorDenied(t *testing.T, stdout string) bool {
	t.Helper()
	var out cursorOutput
	require.NoError(t, json.Unmarshal([]byte(stdout), &out), "cursor always gets JSON: %q", stdout)
	if out.Permission == permissionAllow {
		return false
	}
	assert.Equal(t, permissionDeny, out.Permission)
	assert.Contains(t, out.AgentMessage, "ast-context-cache")
	return true
}

func TestSearchNudge(t *testing.T) {
	sid := "11111111-2222-3333-4444-555555555555"
	grep := map[string]any{"pattern": "func main"}
	goRead := map[string]any{"file_path": "/repo/main.go"}
	type step struct {
		event    string
		payload  func(t *testing.T, transcript string) []byte
		wantDeny bool
	}
	claude := func(tool string, in map[string]any, deny bool) step {
		return step{event: EventSearchNudge, wantDeny: deny, payload: func(t *testing.T, tr string) []byte { return toolPayload(t, sid, "", tool, tr, in) }}
	}
	tests := []struct {
		name       string
		flag       *bool
		down       bool
		env        string
		transcript string
		steps      []step
		wantHealth int32
	}{
		{name: "first grep blocked, second allowed", flag: boolPtr(true), steps: []step{claude("Grep", grep, true), claude("Grep", grep, false), claude("Glob", grep, false)}, wantHealth: 1},
		{name: "first glob blocked", flag: boolPtr(true), steps: []step{claude("Glob", map[string]any{"pattern": "**/*.go"}, true), claude("Read", goRead, false)}, wantHealth: 1},
		{name: "whole-file code read blocked", flag: boolPtr(true), steps: []step{claude("Read", goRead, true), claude("Read", goRead, false)}, wantHealth: 1},
		{
			name: "prior ast call in transcript allows", flag: boolPtr(true),
			transcript: `{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_0","name":"mcp__ast-context-cache__get_project_map","input":{}}]}}` + "\n",
			steps:      []step{claude("Grep", grep, false), claude("Read", goRead, false)},
		},
		{
			name: "gateway-prefixed ast call allows", flag: boolPtr(true),
			transcript: `{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1","name":"mcp__mcp-local__ast_search_semantic","input":{}}]}}` + "\n",
			steps:      []step{claude("Grep", grep, false)},
		},
		{
			name: "other mcp calls in transcript do not count", flag: boolPtr(true),
			transcript: `{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1","name":"mcp__github__search_code","input":{}}]}}` + "\n",
			steps:      []step{claude("Grep", grep, true)}, wantHealth: 1,
		},
		{
			name: "ast tool definitions in transcript do not count", flag: boolPtr(true),
			transcript: `{"type":"attachment","tools":[{"name":"mcp__ast-context-cache__get_context_capsule","description":"Search","input_schema":{},"defer_loading":true}]}` + "\n",
			steps:      []step{claude("Grep", grep, true)}, wantHealth: 1,
		},
		{name: "ast tool call marks the session", flag: boolPtr(true), steps: []step{claude("mcp__ast-context-cache__get_context_capsule", map[string]any{"query": "x"}, false), claude("Grep", grep, false)}},
		{
			name: "non-code and partial reads allowed and keep the nudge", flag: boolPtr(true),
			steps: []step{
				claude("Read", map[string]any{"file_path": "/repo/README.md"}, false),
				claude("Read", map[string]any{"file_path": "/repo/config.yaml"}, false),
				claude("Read", map[string]any{"file_path": "/repo/logo.png"}, false),
				claude("Read", map[string]any{"file_path": "/repo/main.go", "offset": 10, "limit": 40}, false),
				claude("Bash", map[string]any{"command": "grep -r x ."}, false),
				claude("Grep", grep, true),
			},
			wantHealth: 1,
		},
		{name: "flag off on server allows", flag: boolPtr(false), steps: []step{claude("Grep", grep, false), claude("Grep", grep, false)}, wantHealth: 2},
		{name: "server without the flag allows", steps: []step{claude("Grep", grep, false)}, wantHealth: 1},
		{name: "env off allows without asking the server", flag: boolPtr(true), env: "false", steps: []step{claude("Grep", grep, false)}},
		{name: "server down allows", down: true, steps: []step{claude("Grep", grep, false)}},
		{
			name: "malformed input allows", flag: boolPtr(true),
			steps: []step{
				{event: EventSearchNudge, payload: func(*testing.T, string) []byte { return []byte("nope") }},
				{event: EventSearchNudge, payload: func(t *testing.T, _ string) []byte { return encode(t, map[string]any{"tool_name": "Grep"}) }},
			},
		},
		{
			name: "subagent gets its own nudge", flag: boolPtr(true),
			steps: []step{
				claude("Grep", grep, true),
				{event: EventSearchNudge, wantDeny: true, payload: func(t *testing.T, tr string) []byte { return toolPayload(t, sid, "agent-1", "Grep", tr, grep) }},
				{event: EventSearchNudge, payload: func(t *testing.T, tr string) []byte { return toolPayload(t, sid, "agent-1", "Grep", tr, grep) }},
			},
			wantHealth: 2,
		},
		{
			name: "cursor read blocked once", flag: boolPtr(true),
			steps: []step{
				{event: EventCursorSearchNudge, wantDeny: true, payload: func(t *testing.T, _ string) []byte {
					return cursorPayload(t, "conv-1", "Read", map[string]any{"path": "/repo/a.ts"})
				}},
				{event: EventCursorSearchNudge, payload: func(t *testing.T, _ string) []byte { return cursorPayload(t, "conv-1", "Grep", grep) }},
			},
			wantHealth: 1,
		},
		{
			name: "cursor ast tool marks the conversation", flag: boolPtr(true),
			steps: []step{
				{event: EventCursorSearchNudge, payload: func(t *testing.T, _ string) []byte { return cursorPayload(t, "conv-2", "MCP:get_file_context", nil) }},
				{event: EventCursorSearchNudge, payload: func(t *testing.T, _ string) []byte { return cursorPayload(t, "conv-2", "Grep", grep) }},
			},
		},
		{
			name: "cursor malformed input allows with JSON", flag: boolPtr(true),
			steps: []step{{event: EventCursorSearchNudge, payload: func(*testing.T, string) []byte { return []byte("{") }}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(NudgeEnv, tt.env)
			url, hits := fakeHealth(t, tt.flag)
			if tt.down {
				url = "http://127.0.0.1:1/mcp"
			}
			dir := t.TempDir()
			transcript := ""
			if tt.transcript != "" {
				transcript = filepath.Join(dir, "session.jsonl")
				require.NoError(t, os.WriteFile(transcript, []byte(tt.transcript), 0o600))
			}
			h := New(url, filepath.Join(dir, "hooks"))
			for i, s := range tt.steps {
				out := run(t, h, s.event, s.payload(t, transcript))
				got := false
				if s.event == EventCursorSearchNudge {
					got = cursorDenied(t, out)
				} else {
					got = denied(t, out)
				}
				assert.Equal(t, s.wantDeny, got, "step %d: %q", i, out)
			}
			assert.Equal(t, tt.wantHealth, hits.Load(), "health checks")
		})
	}
}

func TestSearchNudgeParallelBlocksOnce(t *testing.T) {
	url, _ := fakeHealth(t, boolPtr(true))
	h := New(url, filepath.Join(t.TempDir(), "hooks"))
	payload := toolPayload(t, "parallel-session", "", "Grep", "", map[string]any{"pattern": "x"})
	const n = 8
	results := make(chan bool, n)
	for range n {
		go func() { results <- run(t, h, EventSearchNudge, payload) != "" }()
	}
	blocked := 0
	for range n {
		if <-results {
			blocked++
		}
	}
	assert.Equal(t, 1, blocked)
}

func TestNudgeKind(t *testing.T) {
	tests := []struct {
		tool  string
		input string
		want  string
	}{
		{"Grep", `{"pattern":"x"}`, NudgeGrep},
		{"grep", `{"pattern":"x"}`, NudgeGrep},
		{"Glob", `{"pattern":"*.go"}`, NudgeGlob},
		{"Read", `{"file_path":"/a/b.go"}`, NudgeRead},
		{"read", `{"filePath":"/a/b.TS"}`, NudgeRead},
		{"Read", `{"target_file":"x.py"}`, NudgeRead},
		{"Read", `{"file_path":"/a/b.go","offset":null}`, NudgeRead},
		{"Read", `{"file_path":"/a/b.go","limit":20}`, ""},
		{"Read", `{"file_path":"/a/b.go","offset":3}`, ""},
		{"Read", `{"file_path":"/a/b.md"}`, ""},
		{"Read", `{"file_path":"/a/package.json"}`, ""},
		{"Read", `{"file_path":"/a/Makefile"}`, ""},
		{"Read", `{}`, ""},
		{"Bash", `{"command":"rg x"}`, ""},
		{"Edit", `{"file_path":"/a/b.go"}`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.tool+tt.input, func(t *testing.T) {
			var in map[string]json.RawMessage
			require.NoError(t, json.Unmarshal([]byte(tt.input), &in))
			assert.Equal(t, tt.want, NudgeKind(tt.tool, in))
		})
	}
}

func TestIsASTTool(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"mcp__ast-context-cache__get_context_capsule", true},
		{"mcp__ast-context-cache__handoff", true},
		{"mcp__plugin_ast_context_cache__recall_memory", true},
		{"mcp__gateway__ast_get_file_context", true},
		{"mcp__mcp-local__ast-context-cache-search_semantic", true},
		{"MCP:get_project_map", true},
		{"MCP:search_semantic", true},
		{"MCP:create_issue", false},
		{"mcp__github__search_code", false},
		{"get_context_capsule", false},
		{"Grep", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, IsASTTool(tt.name))
		})
	}
}

func TestHealthURL(t *testing.T) {
	tests := []struct{ in, want string }{
		{"http://127.0.0.1:7821/mcp", "http://127.0.0.1:7821/health"},
		{"http://127.0.0.1:7821/mcp/", "http://127.0.0.1:7821/health"},
		{"https://ast.example.ts.net/prefix/mcp?x=1", "https://ast.example.ts.net/prefix/health"},
		{"not a url", ""},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, HealthURL(tt.in), tt.in)
	}
}
