package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func hookFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "docs", "spikes", "fixtures", name+".json"))
	require.NoError(t, err)
	return data
}

func runHookTest(t *testing.T, stdin []byte, args ...string) (int, string, string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code := runHook(args, bytes.NewReader(stdin), &out, &errOut)
	return code, out.String(), errOut.String()
}

// downURL is an MCP URL nothing listens on.
func downURL(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL + "/mcp"
	srv.Close()
	return url
}

func TestCLIHookSessionStart(t *testing.T) {
	cliHome(t)
	t.Setenv("AST_MCP_URL", downURL(t))
	t.Setenv("AST_HOOK_DEBUG", "")
	code, out, errOut := runHookTest(t, hookFixture(t, "SessionStart.startup"), "session-start")
	assert.Equal(t, exitOK, code)
	assert.Empty(t, errOut)
	var resp map[string]map[string]string
	require.NoError(t, json.Unmarshal([]byte(out), &resp))
	assert.Equal(t, "SessionStart", resp["hookSpecificOutput"]["hookEventName"])
	assert.Contains(t, resp["hookSpecificOutput"]["additionalContext"], "session_id=82de5123-70a6-4982-b7cc-aa839be2cd89")
}

func TestCLIHookFailsOpen(t *testing.T) {
	cliHome(t)
	t.Setenv("AST_MCP_URL", downURL(t))
	tests := []struct {
		name  string
		debug string
		stdin []byte
		args  []string
	}{
		{name: "server down", stdin: hookFixture(t, "PreToolUse.Agent"), args: []string{"pre-tool-use-agent"}},
		{name: "server down with debug", debug: "1", stdin: hookFixture(t, "PreToolUse.Agent"), args: []string{"pre-tool-use-agent"}},
		{name: "malformed stdin", stdin: []byte("nope"), args: []string{"subagent-start"}},
		{name: "no event", stdin: hookFixture(t, "Stop"), args: nil},
		{name: "unknown event", stdin: hookFixture(t, "Stop"), args: []string{"stop"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("AST_HOOK_DEBUG", tt.debug)
			start := time.Now()
			code, out, errOut := runHookTest(t, tt.stdin, tt.args...)
			assert.Less(t, time.Since(start), 2500*time.Millisecond)
			assert.Equal(t, exitOK, code)
			assert.Empty(t, out)
			if tt.debug == "" {
				assert.Empty(t, errOut, "diagnostics only with AST_HOOK_DEBUG=1")
				return
			}
			assert.True(t, strings.Contains(errOut, "Hook produced no output"), errOut)
		})
	}
}

func TestCLIHookSearchNudge(t *testing.T) {
	cliHome(t)
	t.Setenv("AST_HOOK_DEBUG", "")
	t.Setenv("AST_FEATURE_SEARCH_NUDGE", "")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"status":"healthy","search_nudge":true}`))
	}))
	t.Cleanup(srv.Close)
	t.Setenv("AST_MCP_URL", srv.URL+"/mcp")
	grep := []byte(`{"session_id":"cli-nudge","hook_event_name":"PreToolUse","tool_name":"Grep","tool_input":{"pattern":"func main"}}`)
	code, out, errOut := runHookTest(t, grep, "search-nudge")
	assert.Equal(t, exitOK, code)
	assert.Empty(t, errOut)
	var resp map[string]map[string]string
	require.NoError(t, json.Unmarshal([]byte(out), &resp))
	assert.Equal(t, "deny", resp["hookSpecificOutput"]["permissionDecision"])
	assert.Contains(t, resp["hookSpecificOutput"]["permissionDecisionReason"], "get_context_capsule")
	_, out, _ = runHookTest(t, grep, "search-nudge")
	assert.Empty(t, out, "the nudge blocks once per session")
}

func TestCLIHookSearchNudgeFailsOpen(t *testing.T) {
	cliHome(t)
	t.Setenv("AST_HOOK_DEBUG", "")
	t.Setenv("AST_MCP_URL", downURL(t))
	grep := []byte(`{"session_id":"cli-nudge-down","conversation_id":"cli-nudge-down","tool_name":"Grep","tool_input":{"pattern":"x"}}`)
	tests := []struct {
		name, event string
		stdin       []byte
		want        string
	}{
		{name: "claude server down", event: "search-nudge", stdin: grep},
		{name: "claude malformed", event: "search-nudge", stdin: []byte("nope")},
		{name: "cursor server down", event: "cursor-search-nudge", stdin: grep, want: `{"permission":"allow"}` + "\n"},
		{name: "cursor malformed", event: "cursor-search-nudge", stdin: []byte("nope"), want: `{"permission":"allow"}` + "\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, out, errOut := runHookTest(t, tt.stdin, tt.event)
			assert.Equal(t, exitOK, code)
			assert.Equal(t, tt.want, out)
			assert.Empty(t, errOut)
		})
	}
}
