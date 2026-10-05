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
