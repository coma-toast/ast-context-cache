package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/handoff"
)

const (
	parentSID = "82de5123-70a6-4982-b7cc-aa839be2cd89"
	refA      = "hof_00000000000000aa"
	refB      = "hof_00000000000000bb"
	// failOpenBudget is how long a failing hook may take (HI-5: 2s timeout plus slack).
	failOpenBudget = 2500 * time.Millisecond
)

// toolCall is one tools/call the fake server received.
type toolCall struct {
	Tool string
	Args map[string]any
}

// fakeMCP is an httptest MCP server that records tools/call requests and answers each with
// reply's result, wrapped the way the real server wraps it.
type fakeMCP struct {
	mu    sync.Mutex
	calls []toolCall
	reply func(c toolCall) (result any, isError bool)
	srv   *httptest.Server
}

func newFakeMCP(t *testing.T, reply func(c toolCall) (any, bool)) *fakeMCP {
	t.Helper()
	f := &fakeMCP{reply: reply}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     int    `json:"id"`
			Method string `json:"method"`
			Params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"params"`
		}
		if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&req)) {
			return
		}
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "tools/call", req.Method)
		c := toolCall{Tool: req.Params.Name, Args: req.Params.Arguments}
		f.mu.Lock()
		f.calls = append(f.calls, c)
		f.mu.Unlock()
		result, isError := f.reply(c)
		text, err := json.Marshal(result)
		assert.NoError(t, err)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0", "id": req.ID,
			"result": map[string]any{"content": []map[string]any{{"type": "text", "text": string(text)}}, "isError": isError},
		})
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeMCP) Calls() []toolCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]toolCall(nil), f.calls...)
}

func (f *fakeMCP) URL() string {
	return f.srv.URL + "/mcp"
}

// noCalls answers every call with an error; tests that expect no calls use it.
func noCalls(t *testing.T) func(toolCall) (any, bool) {
	return func(c toolCall) (any, bool) {
		t.Errorf("unexpected tool call %s %v", c.Tool, c.Args)
		return map[string]string{"error": "unexpected"}, true
	}
}

func fixture(t *testing.T, name string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "docs", "spikes", "fixtures", name+".json"))
	require.NoError(t, err)
	var m map[string]any
	require.NoError(t, json.Unmarshal(data, &m))
	return m
}

func encode(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return b
}

// run runs one hook event and returns its stdout.
func run(t *testing.T, h *Handler, event string, payload []byte) string {
	t.Helper()
	var out bytes.Buffer
	h.Run(context.Background(), event, bytes.NewReader(payload), &out)
	return out.String()
}

// decodeOutput parses hook stdout, which must be a single JSON object line.
func decodeOutput(t *testing.T, stdout string) Output {
	t.Helper()
	require.True(t, strings.HasSuffix(stdout, "}\n"), "stdout is one JSON line: %q", stdout)
	var out Output
	require.NoError(t, json.Unmarshal([]byte(stdout), &out))
	return out
}

func TestSessionStart(t *testing.T) {
	want := "ast-context-cache: use session_id=" + parentSID + " for all ast-context-cache tool calls in this conversation."
	list := handoff.ListResponse{Handoffs: []handoff.HandoffSummary{
		{Ref: refB, Label: "Review auth", Children: 2, StatusCounts: map[handoff.Status]int{handoff.StatusOpen: 1, handoff.StatusDone: 1}},
		{Ref: refA, Label: "", Children: 0},
	}}
	tests := []struct {
		name     string
		fixture  string
		reply    func(t *testing.T) func(toolCall) (any, bool)
		wantText string
		wantList bool
	}{
		{name: "startup", fixture: "SessionStart.startup", reply: noCalls, wantText: want},
		{name: "resume", fixture: "SessionStart.resume", reply: noCalls, wantText: want},
		{
			name: "compact lists handoffs", fixture: "SessionStart.compact", wantList: true,
			reply: func(t *testing.T) func(toolCall) (any, bool) {
				return func(toolCall) (any, bool) { return list, false }
			},
			wantText: want + "\nHandoffs this session created before compaction (newest first):\n" +
				"- " + refB + ` "Review auth": 1 open, 1 done` + "\n" +
				"- " + refA + ": no child opened yet\n" +
				"Use `handoff` action `collect` with session_id=" + parentSID + " to read results.",
		},
		{
			name: "compact with list failing still injects the session id", fixture: "SessionStart.compact", wantList: true,
			reply: func(t *testing.T) func(toolCall) (any, bool) {
				return func(toolCall) (any, bool) { return map[string]string{"error": "boom"}, true }
			},
			wantText: want,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeMCP(t, tt.reply(t))
			h := New(srv.URL(), t.TempDir())
			out := decodeOutput(t, run(t, h, EventSessionStart, encode(t, fixture(t, tt.fixture))))
			assert.Equal(t, "SessionStart", out.HookSpecificOutput.HookEventName)
			assert.Equal(t, tt.wantText, out.HookSpecificOutput.AdditionalContext)
			if !tt.wantList {
				assert.Empty(t, srv.Calls())
				return
			}
			require.Len(t, srv.Calls(), 1)
			assert.Equal(t, toolCall{Tool: "handoff", Args: map[string]any{"action": "list", "session_id": parentSID}}, srv.Calls()[0])
		})
	}
}

func TestSessionStartListCapped(t *testing.T) {
	var hs []handoff.HandoffSummary
	for range 25 {
		hs = append(hs, handoff.HandoffSummary{
			Ref: refA, Label: strings.Repeat("long label ", 20), Children: 3,
			StatusCounts: map[handoff.Status]int{handoff.StatusOpen: 1, handoff.StatusDone: 1, handoff.StatusPartial: 1},
		})
	}
	srv := newFakeMCP(t, func(toolCall) (any, bool) { return handoff.ListResponse{Handoffs: hs}, false })
	out := decodeOutput(t, run(t, New(srv.URL(), t.TempDir()), EventSessionStart, encode(t, fixture(t, "SessionStart.compact"))))
	text := out.HookSpecificOutput.AdditionalContext
	assert.Contains(t, text, "+15 more")
	assert.LessOrEqual(t, len(text)/charsPerToken, 450, "about 400 tokens")
}

func openDigest(ref string, child string) handoff.OpenResponse {
	return handoff.OpenResponse{
		Handoff: handoff.HandoffRef(ref), SessionID: handoff.SessionID(child), Mode: handoff.ModeFresh, Label: "Report canaries",
		Brief:    "Report canaries\n\nFind the canary.",
		Pointers: []handoff.PointerDigest{{ID: 7, Key: "internal/hooks/hooks.go|Run", Note: "entry point"}},
		Trail:    []handoff.TrailDigest{{ID: 3, Tool: "search_semantic", Query: "canary", Hits: 4}},
	}
}

func TestSubagentStart(t *testing.T) {
	generic := "ast-context-cache: the parent conversation's session_id is " + parentSID + ". If your prompt contains " +
		"[handoff hof_…], call open_handoff first and use the child session_id it returns for all ast-context-cache calls."
	t.Run("general-purpose without a pending handoff gets the parent session id", func(t *testing.T) {
		srv := newFakeMCP(t, noCalls(t))
		out := decodeOutput(t, run(t, New(srv.URL(), t.TempDir()), EventSubagentStart, encode(t, fixture(t, "SubagentStart.general-purpose"))))
		assert.Equal(t, "SubagentStart", out.HookSpecificOutput.HookEventName)
		assert.Equal(t, generic, out.HookSpecificOutput.AdditionalContext)
	})
	t.Run("fork gets nothing", func(t *testing.T) {
		srv := newFakeMCP(t, noCalls(t))
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.AddPending(context.Background(), "776dc3f7-daed-4f4b-a8f2-8976cc28bd51", PendingHandoff{Ref: refA}))
		assert.Empty(t, run(t, h, EventSubagentStart, encode(t, fixture(t, "SubagentStart.fork"))))
		_, ok, err := h.registry.TakePending(context.Background(), "776dc3f7-daed-4f4b-a8f2-8976cc28bd51")
		require.NoError(t, err)
		assert.True(t, ok, "a fork leaves the pending handoff queued")
	})
	t.Run("pending handoffs are opened oldest first", func(t *testing.T) {
		srv := newFakeMCP(t, func(c toolCall) (any, bool) {
			ref := c.Args["handoff"].(string)
			return openDigest(ref, ref+".c1"), false
		})
		h := New(srv.URL(), t.TempDir())
		ctx := context.Background()
		require.NoError(t, h.registry.AddPending(ctx, parentSID, PendingHandoff{Ref: refA}))
		require.NoError(t, h.registry.AddPending(ctx, parentSID, PendingHandoff{Ref: refB}))
		in := fixture(t, "SubagentStart.general-purpose")
		for i, ref := range []string{refA, refB} {
			in["agent_id"] = "agent" + ref
			out := decodeOutput(t, run(t, h, EventSubagentStart, encode(t, in)))
			text := out.HookSpecificOutput.AdditionalContext
			assert.Contains(t, text, "this subagent is handoff "+ref)
			assert.Contains(t, text, "Use session_id="+ref+".c1 for all ast-context-cache calls")
			assert.Contains(t, text, "Brief: Report canaries\n\nFind the canary.")
			assert.Contains(t, text, "- 7 internal/hooks/hooks.go|Run — entry point")
			assert.Contains(t, text, `- search_semantic "canary" (4 hits)`)
			calls := srv.Calls()
			require.Len(t, calls, i+1)
			assert.Equal(t, toolCall{Tool: "open_handoff", Args: map[string]any{"action": "open", "handoff": ref, "project_path": "/tmp/hook-spike/work"}}, calls[i])
			a, ok, err := h.registry.Agent(ctx, parentSID, "agent"+ref)
			require.NoError(t, err)
			require.True(t, ok)
			assert.Equal(t, AgentHandoff{Ref: ref, SessionID: ref + ".c1", CreatedAt: a.CreatedAt}, a)
		}
		_, ok, err := h.registry.TakePending(ctx, parentSID)
		require.NoError(t, err)
		assert.False(t, ok, "both pending handoffs were consumed")
	})
	t.Run("the stub in the subagent transcript wins over the queue", func(t *testing.T) {
		srv := newFakeMCP(t, func(c toolCall) (any, bool) { return openDigest(refB, refB+".c1"), false })
		h := New(srv.URL(), t.TempDir())
		ctx := context.Background()
		require.NoError(t, h.registry.AddPending(ctx, parentSID, PendingHandoff{Ref: refA}))
		require.NoError(t, h.registry.AddPending(ctx, parentSID, PendingHandoff{Ref: refB}))
		in := fixture(t, "SubagentStart.general-purpose")
		dir := t.TempDir()
		in["transcript_path"] = filepath.Join(dir, parentSID+".jsonl")
		sub := filepath.Join(dir, parentSID, "subagents", "agent-ae37e4a6c4dca43fd.jsonl")
		require.NoError(t, os.MkdirAll(filepath.Dir(sub), 0o700))
		require.NoError(t, os.WriteFile(sub, []byte(`{"type":"user","message":{"content":"do it\n\n[handoff `+refB+`] Report — call open_handoff first"}}`+"\n"), 0o600))
		decodeOutput(t, run(t, h, EventSubagentStart, encode(t, in)))
		require.Len(t, srv.Calls(), 1)
		assert.Equal(t, refB, srv.Calls()[0].Args["handoff"])
		p, ok, err := h.registry.TakePending(ctx, parentSID)
		require.NoError(t, err)
		require.True(t, ok)
		assert.Equal(t, refA, p.Ref, "the other subagent's handoff stays queued")
	})
	t.Run("open failing falls back to the generic text", func(t *testing.T) {
		srv := newFakeMCP(t, func(toolCall) (any, bool) { return map[string]string{"error": "handoff not found"}, true })
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.AddPending(context.Background(), parentSID, PendingHandoff{Ref: refA}))
		out := decodeOutput(t, run(t, h, EventSubagentStart, encode(t, fixture(t, "SubagentStart.general-purpose"))))
		assert.Equal(t, generic, out.HookSpecificOutput.AdditionalContext)
	})
	t.Run("digest is capped", func(t *testing.T) {
		srv := newFakeMCP(t, func(toolCall) (any, bool) {
			d := openDigest(refA, refA+".c1")
			d.Brief = strings.Repeat("x", 20000)
			return d, false
		})
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.AddPending(context.Background(), parentSID, PendingHandoff{Ref: refA}))
		out := decodeOutput(t, run(t, h, EventSubagentStart, encode(t, fixture(t, "SubagentStart.general-purpose"))))
		assert.Less(t, len(out.HookSpecificOutput.AdditionalContext), (maxDigestTokens+120)*charsPerToken)
		assert.True(t, strings.HasSuffix(out.HookSpecificOutput.AdditionalContext, "…"))
	})
}

// stopInput is the general-purpose SubagentStop fixture with its transcript pointed at a real
// file holding transcript.
func stopInput(t *testing.T, transcript string) map[string]any {
	t.Helper()
	in := fixture(t, "SubagentStop.general-purpose")
	path := filepath.Join(t.TempDir(), "agent-ae37e4a6c4dca43fd.jsonl")
	require.NoError(t, os.WriteFile(path, []byte(transcript), 0o600))
	in["agent_transcript_path"] = path
	return in
}

func TestSubagentStop(t *testing.T) {
	const child = refA + ".c1"
	ctx := context.Background()
	collect := func(status handoff.Status) handoff.CollectResponse {
		return handoff.CollectResponse{Children: []handoff.ChildResult{
			{SessionID: refA + ".c2", Handoff: refA, Status: handoff.StatusOpen},
			{SessionID: child, Handoff: refA, Status: status},
		}}
	}
	reply := func(status handoff.Status) func(c toolCall) (any, bool) {
		return func(c toolCall) (any, bool) {
			if c.Args["action"] == "collect" {
				return collect(status), false
			}
			return handoff.CompleteResponse{ResultRef: "ctx_1", Handoff: refA, Status: handoff.StatusPartial}, false
		}
	}
	t.Run("compaction summarizer is ignored", func(t *testing.T) {
		srv := newFakeMCP(t, noCalls(t))
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.SetAgent(ctx, parentSID, "a7152e5968941ab82", AgentHandoff{Ref: refA, SessionID: child}))
		assert.Empty(t, run(t, h, EventSubagentStop, encode(t, fixture(t, "SubagentStop.compaction-summarizer"))))
	})
	t.Run("missing transcript is ignored", func(t *testing.T) {
		srv := newFakeMCP(t, noCalls(t))
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.SetAgent(ctx, parentSID, "ae37e4a6c4dca43fd", AgentHandoff{Ref: refA, SessionID: child}))
		assert.Empty(t, run(t, h, EventSubagentStop, encode(t, fixture(t, "SubagentStop.general-purpose"))))
	})
	t.Run("subagent without a hook-opened handoff is ignored", func(t *testing.T) {
		srv := newFakeMCP(t, noCalls(t))
		assert.Empty(t, run(t, New(srv.URL(), t.TempDir()), EventSubagentStop, encode(t, stopInput(t, ""))))
	})
	t.Run("open child is completed as partial with the last message", func(t *testing.T) {
		srv := newFakeMCP(t, reply(handoff.StatusOpen))
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.SetAgent(ctx, parentSID, "ae37e4a6c4dca43fd", AgentHandoff{Ref: refA, SessionID: child}))
		assert.Empty(t, run(t, h, EventSubagentStop, encode(t, stopInput(t, ""))))
		assert.Equal(t, []toolCall{
			{Tool: "handoff", Args: map[string]any{"action": "collect", "handoff": refA}},
			{Tool: "handoff", Args: map[string]any{
				"action": "complete", "session_id": child, "status": "partial", "project_path": "/tmp/hook-spike/work",
				"content": "Subagent canary: CANARY-SUBAGENT-4402\nSession canary: NONE\nCANARY-PROMPT-9915",
			}},
		}, srv.Calls())
		_, ok, err := h.registry.Agent(ctx, parentSID, "ae37e4a6c4dca43fd")
		require.NoError(t, err)
		assert.False(t, ok, "the agent mapping is consumed")
	})
	t.Run("empty last message falls back to the transcript tail", func(t *testing.T) {
		srv := newFakeMCP(t, reply(handoff.StatusOpen))
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.SetAgent(ctx, parentSID, "ae37e4a6c4dca43fd", AgentHandoff{Ref: refA, SessionID: child}))
		in := stopInput(t, `{"type":"assistant","message":{"content":[{"type":"text","text":"first"}]}}`+"\n"+
			`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read"},{"type":"text","text":"found half of it"}]}}`+"\n"+
			`{"type":"user","message":{"content":"tool result"}}`+"\n")
		in["last_assistant_message"] = ""
		run(t, h, EventSubagentStop, encode(t, in))
		calls := srv.Calls()
		require.Len(t, calls, 2)
		assert.Equal(t, "found half of it", calls[1].Args["content"])
	})
	t.Run("completed child is left alone", func(t *testing.T) {
		srv := newFakeMCP(t, reply(handoff.StatusDone))
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.SetAgent(ctx, parentSID, "ae37e4a6c4dca43fd", AgentHandoff{Ref: refA, SessionID: child}))
		assert.Empty(t, run(t, h, EventSubagentStop, encode(t, stopInput(t, ""))))
		require.Len(t, srv.Calls(), 1)
		assert.Equal(t, "collect", srv.Calls()[0].Args["action"])
	})
}

func TestPreToolUseAgent(t *testing.T) {
	stub := "[handoff " + refA + "] Report canaries — call open_handoff first"
	created := func(toolCall) (any, bool) {
		return handoff.CreateResponse{Ref: refA, TreeID: "hft_00000000000000aa", Stub: stub}, false
	}
	t.Run("creates a handoff and appends its stub", func(t *testing.T) {
		srv := newFakeMCP(t, created)
		h := New(srv.URL(), t.TempDir())
		in := fixture(t, "PreToolUse.Agent")
		toolInput := in["tool_input"].(map[string]any)
		toolInput["model"] = "haiku"
		toolInput["run_in_background"] = true
		toolInput["max_turns"] = 12345678901234567
		stdout := run(t, h, EventPreToolUseAgent, encode(t, in))
		assert.NotContains(t, stdout, "permissionDecision")
		var raw map[string]map[string]json.RawMessage
		require.NoError(t, json.Unmarshal([]byte(stdout), &raw))
		assert.Equal(t, `"PreToolUse"`, string(raw["hookSpecificOutput"]["hookEventName"]))
		var updated map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw["hookSpecificOutput"]["updatedInput"], &updated))
		assert.Equal(t, map[string]json.RawMessage{
			"description":       json.RawMessage(`"Report canaries"`),
			"prompt":            encode(t, toolInput["prompt"].(string)+"\n\n"+stub),
			"subagent_type":     json.RawMessage(`"general-purpose"`),
			"model":             json.RawMessage(`"haiku"`),
			"run_in_background": json.RawMessage(`true`),
			"max_turns":         json.RawMessage(`12345678901234567`),
		}, updated)
		require.Len(t, srv.Calls(), 1)
		assert.Equal(t, toolCall{Tool: "handoff", Args: map[string]any{
			"action": "create", "session_id": parentSID, "label": "Report canaries", "project_path": "/tmp/hook-spike/work",
			"brief": "Report canaries\n\n" + toolInput["prompt"].(string),
		}}, srv.Calls()[0])
		p, ok, err := h.registry.TakePending(context.Background(), parentSID)
		require.NoError(t, err)
		require.True(t, ok)
		assert.Equal(t, refA, p.Ref)
		assert.Equal(t, "toolu_01UDto3QG8vcCwa2XBPaKX2T", p.ToolUseID)
	})
	t.Run("brief takes only the head of a long prompt", func(t *testing.T) {
		srv := newFakeMCP(t, created)
		in := fixture(t, "PreToolUse.Agent")
		in["tool_input"].(map[string]any)["prompt"] = strings.Repeat("é", 5000)
		run(t, New(srv.URL(), t.TempDir()), EventPreToolUseAgent, encode(t, in))
		require.Len(t, srv.Calls(), 1)
		assert.Equal(t, "Report canaries\n\n"+strings.Repeat("é", maxBriefPromptRunes-1)+"…", srv.Calls()[0].Args["brief"])
	})
	skips := []struct {
		name   string
		mutate func(in map[string]any)
	}{
		{name: "fork", mutate: func(in map[string]any) { in["tool_input"].(map[string]any)["subagent_type"] = "fork" }},
		{name: "prompt already has a stub", mutate: func(in map[string]any) {
			in["tool_input"].(map[string]any)["prompt"] = "go\n\n[handoff " + refB + "] x — call open_handoff first"
		}},
		{name: "other tool", mutate: func(in map[string]any) { in["tool_name"] = "Read" }},
		{name: "unknown subagent spawning", mutate: func(in map[string]any) { in["agent_id"] = "a0a0899bd5e5ba186" }},
	}
	for _, tt := range skips {
		t.Run(tt.name+" is left alone", func(t *testing.T) {
			srv := newFakeMCP(t, noCalls(t))
			in := fixture(t, "PreToolUse.Agent")
			tt.mutate(in)
			assert.Empty(t, run(t, New(srv.URL(), t.TempDir()), EventPreToolUseAgent, encode(t, in)))
		})
	}
	t.Run("subagent with a hook-opened handoff creates from its child session", func(t *testing.T) {
		srv := newFakeMCP(t, created)
		h := New(srv.URL(), t.TempDir())
		require.NoError(t, h.registry.SetAgent(context.Background(), parentSID, "ae37e4a6c4dca43fd", AgentHandoff{Ref: refB, SessionID: refB + ".c1"}))
		in := fixture(t, "PreToolUse.Agent")
		in["agent_id"] = "ae37e4a6c4dca43fd"
		decodeOutput(t, run(t, h, EventPreToolUseAgent, encode(t, in)))
		require.Len(t, srv.Calls(), 1)
		assert.Equal(t, refB+".c1", srv.Calls()[0].Args["session_id"])
	})
	t.Run("create failing produces no output", func(t *testing.T) {
		srv := newFakeMCP(t, func(toolCall) (any, bool) {
			return map[string]any{"error": "handoff create is turned off by a feature flag", "code": "feature_disabled"}, true
		})
		h := New(srv.URL(), t.TempDir())
		assert.Empty(t, run(t, h, EventPreToolUseAgent, encode(t, fixture(t, "PreToolUse.Agent"))))
		_, ok, err := h.registry.TakePending(context.Background(), parentSID)
		require.NoError(t, err)
		assert.False(t, ok)
	})
}

func TestFailOpen(t *testing.T) {
	down := httptest.NewServer(http.NotFoundHandler())
	downURL := down.URL + "/mcp"
	down.Close()
	release := make(chan struct{})
	hang := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(hang.Close)
	t.Cleanup(func() { close(release) })
	notJSON := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("<html>")) }))
	t.Cleanup(notJSON.Close)
	forbidden := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusForbidden) }))
	t.Cleanup(forbidden.Close)
	agent := encode(t, fixture(t, "PreToolUse.Agent"))
	stopped, stdinOpen := io.Pipe()
	t.Cleanup(func() { _ = stdinOpen.Close() })
	tests := []struct {
		name  string
		url   string
		event string
		stdin io.Reader
	}{
		{name: "server down", url: downURL, event: EventPreToolUseAgent, stdin: bytes.NewReader(agent)},
		{name: "server hangs", url: hang.URL + "/mcp", event: EventPreToolUseAgent, stdin: bytes.NewReader(agent)},
		{name: "server answers garbage", url: notJSON.URL, event: EventPreToolUseAgent, stdin: bytes.NewReader(agent)},
		{name: "server forbids", url: forbidden.URL, event: EventPreToolUseAgent, stdin: bytes.NewReader(agent)},
		{name: "malformed stdin", url: downURL, event: EventSessionStart, stdin: strings.NewReader("{not json")},
		{name: "empty stdin", url: downURL, event: EventSubagentStart, stdin: strings.NewReader("")},
		{name: "stdin never closes", url: downURL, event: EventSessionStart, stdin: stopped},
		{name: "unknown event", url: downURL, event: "post-compact", stdin: bytes.NewReader(agent)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var out bytes.Buffer
			start := time.Now()
			New(tt.url, t.TempDir()).Run(context.Background(), tt.event, tt.stdin, &out)
			assert.Less(t, time.Since(start), failOpenBudget)
			assert.Empty(t, out.String())
		})
	}
}

func TestDefaultURL(t *testing.T) {
	t.Setenv("AST_MCP_URL", "")
	t.Setenv("AST_MCP_PORT", "")
	assert.Equal(t, "http://127.0.0.1:7821/mcp", ResolveURL())
	t.Setenv("AST_MCP_PORT", "9911")
	assert.Equal(t, "http://127.0.0.1:9911/mcp", ResolveURL())
	t.Setenv("AST_MCP_URL", "http://localhost:1234/mcp")
	assert.Equal(t, "http://localhost:1234/mcp", ResolveURL())
}

// TestResultTypesMatchHandoff decodes the real handoff responses into the hooks' mirror types,
// so a renamed JSON field fails here instead of silently emptying a hook's output.
func TestResultTypesMatchHandoff(t *testing.T) {
	var c createResult
	require.NoError(t, json.Unmarshal(encode(t, handoff.CreateResponse{Ref: refA, Stub: "stub"}), &c))
	assert.Equal(t, createResult{Ref: refA, Stub: "stub"}, c)
	var o openResult
	require.NoError(t, json.Unmarshal(encode(t, openDigest(refA, refA+".c1")), &o))
	assert.Equal(t, refA+".c1", o.SessionID)
	assert.Equal(t, "Report canaries", o.Label)
	require.Len(t, o.Pointers, 1)
	assert.Equal(t, "entry point", o.Pointers[0].Note)
	require.Len(t, o.Trail, 1)
	assert.Equal(t, "canary", o.Trail[0].Query)
	var l listResult
	require.NoError(t, json.Unmarshal(encode(t, handoff.ListResponse{Handoffs: []handoff.HandoffSummary{
		{Ref: refA, Label: "x", Children: 1, StatusCounts: map[handoff.Status]int{handoff.StatusOpen: 1}},
	}}), &l))
	assert.Equal(t, listResult{Handoffs: []handoffSummary{{Ref: refA, Label: "x", Children: 1, StatusCounts: map[string]int{"open": 1}}}}, l)
	var col collectResult
	require.NoError(t, json.Unmarshal(encode(t, handoff.CollectResponse{Children: []handoff.ChildResult{{SessionID: "s", Status: handoff.StatusOpen}}}), &col))
	require.Len(t, col.Children, 1)
	assert.Equal(t, "s", col.Children[0].SessionID)
	assert.Equal(t, statusOpen, col.Children[0].Status)
	assert.Equal(t, statusPartial, string(handoff.StatusPartial))
}
