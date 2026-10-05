package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/version"
)

// The CLI opens and closes the global db pools on every call, so these tests don't run in parallel.

func cliHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("DB_PATH", "")
	t.Setenv("AST_MCP_PORT", "")
	return home
}

func runCLITest(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code, handled := runCLIWith(args, &out, &errOut)
	require.True(t, handled)
	return code, out.String(), errOut.String()
}

func sortedKeys(m map[string]json.RawMessage) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func TestCLIVersion(t *testing.T) {
	for _, arg := range []string{"--version", "-version", "version"} {
		code, out, _ := runCLITest(t, arg)
		assert.Equal(t, exitOK, code)
		assert.Equal(t, "ast-mcp "+version.Version+"\n", out)
	}
}

func TestCLINotASubcommand(t *testing.T) {
	_, handled := runCLIWith(nil, &bytes.Buffer{}, &bytes.Buffer{})
	assert.False(t, handled)
	_, handled = runCLIWith([]string{"-tier", "core"}, &bytes.Buffer{}, &bytes.Buffer{})
	assert.False(t, handled, "server flags fall through to the server")
	code, handled := runCLIWith([]string{"hook", "session-start"}, &bytes.Buffer{}, &bytes.Buffer{})
	assert.True(t, handled)
	assert.Equal(t, exitOK, code)
}

func TestCLIInstallExitCodes(t *testing.T) {
	home := cliHome(t)
	mcp := filepath.Join(home, ".cursor", "mcp.json")

	code, out, errOut := runCLITest(t, "install", "--target", "cursor")
	assert.Equal(t, exitConfirm, code, errOut)
	assert.Contains(t, out, "+++ b"+mcp)
	assert.Contains(t, errOut, "--yes")
	assert.NoFileExists(t, mcp)

	code, out, _ = runCLITest(t, "install", "--target", "cursor", "--dry-run")
	assert.Equal(t, exitOK, code)
	assert.Contains(t, out, "not_installed")
	assert.NoFileExists(t, mcp)

	code, _, errOut = runCLITest(t, "install", "--target", "cursor", "--yes")
	assert.Equal(t, exitOK, code, errOut)
	assert.Contains(t, readTestFile(t, mcp), "http://127.0.0.1:7821/mcp")

	code, out, _ = runCLITest(t, "install", "--target", "cursor")
	assert.Equal(t, exitOK, code, "nothing left to confirm")
	assert.Contains(t, out, "already installed")

	code, _, _ = runCLITest(t, "install", "--target", "jetbrains")
	assert.Equal(t, exitUnsupported, code)

	code, _, _ = runCLITest(t, "install", "--target", "cursor,jetbrains", "--dry-run")
	assert.Equal(t, exitUnsupported, code)

	code, _, errOut = runCLITest(t, "install", "--target", "emacs")
	assert.Equal(t, exitError, code)
	assert.Contains(t, errOut, "unknown target")

	code, _, errOut = runCLITest(t, "install")
	assert.Equal(t, exitError, code)
	assert.Contains(t, errOut, "--target is required")
}

func TestCLIParseErrorExit(t *testing.T) {
	home := cliHome(t)
	path := filepath.Join(home, ".codex", "config.toml")
	bad := "[mcp_servers\nurl ="
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(bad), 0o644))
	code, out, _ := runCLITest(t, "install", "--target", "codex", "--yes", "--json")
	assert.Equal(t, exitConflict, code)
	assert.Equal(t, bad, readTestFile(t, path))
	var res cliOutput
	require.NoError(t, json.Unmarshal([]byte(out), &res))
	assert.Contains(t, strings.Join(res.Warnings, "\n"), "could not be parsed")
}

// TestCLIJSONShape pins the --json contract mcp-local depends on.
func TestCLIJSONShape(t *testing.T) {
	home := cliHome(t)
	code, out, errOut := runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--mcp-port", "9911", "--yes", "--json")
	require.Equal(t, exitOK, code, errOut)
	var top map[string]json.RawMessage
	require.NoError(t, json.Unmarshal([]byte(out), &top))
	assert.Equal(t, []string{"changes", "status", "warnings"}, sortedKeys(top))
	var changes []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(top["changes"], &changes))
	require.Len(t, changes, 1)
	assert.Equal(t, []string{"diff", "kind", "path", "reason", "skipped"}, sortedKeys(changes[0]))
	var status []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(top["status"], &status))
	require.NotEmpty(t, status)
	assert.Equal(t, []string{"component", "path", "status", "target"}, sortedKeys(status[0]))
	var res cliOutput
	require.NoError(t, json.Unmarshal([]byte(out), &res))
	assert.Equal(t, "create", res.Changes[0].Kind)
	assert.Equal(t, "installed", res.Status[0].Status)
	assert.Contains(t, readTestFile(t, filepath.Join(home, ".cursor", "mcp.json")), "http://127.0.0.1:9911/mcp")
}

func TestCLIPortFromEnv(t *testing.T) {
	home := cliHome(t)
	t.Setenv("AST_MCP_PORT", "7999")
	code, _, errOut := runCLITest(t, "install", "--target", "opencode", "--component", "mcp", "--yes")
	require.Equal(t, exitOK, code, errOut)
	assert.Contains(t, readTestFile(t, filepath.Join(home, ".config", "opencode", "opencode.json")), "http://127.0.0.1:7999/mcp")
	code, _, _ = runCLITest(t, "install", "--target", "opencode", "--component", "mcp", "--mcp-url", "http://localhost:8000/mcp", "--dry-run")
	assert.Equal(t, exitOK, code)
}

func TestCLIVerifyUninstallBackupsRestore(t *testing.T) {
	home := cliHome(t)
	mcp := filepath.Join(home, ".cursor", "mcp.json")
	require.NoError(t, os.MkdirAll(filepath.Dir(mcp), 0o755))
	require.NoError(t, os.WriteFile(mcp, []byte("{\n  \"mcpServers\": {}\n}\n"), 0o644))
	code, _, errOut := runCLITest(t, "install", "--target", "cursor", "--component", "mcp", "--yes")
	require.Equal(t, exitOK, code, errOut)

	code, out, _ := runCLITest(t, "verify", "--target", "cursor", "--json")
	require.Equal(t, exitOK, code)
	var res cliOutput
	require.NoError(t, json.Unmarshal([]byte(out), &res))
	require.Len(t, res.Status, 4)
	assert.Equal(t, "installed", res.Status[0].Status)
	assert.Empty(t, res.Changes)

	code, out, _ = runCLITest(t, "verify")
	assert.Equal(t, exitOK, code)
	assert.Contains(t, out, "claude_code")

	code, _, errOut = runCLITest(t, "uninstall", "--target", "cursor", "--yes")
	require.Equal(t, exitOK, code, errOut)
	assert.Equal(t, "{\n  \"mcpServers\": {}\n}\n", readTestFile(t, mcp))

	code, out, _ = runCLITest(t, "backups", "--json")
	require.Equal(t, exitOK, code)
	var backups []struct {
		ID   string `json:"id"`
		Path string `json:"path"`
	}
	require.NoError(t, json.Unmarshal([]byte(out), &backups))
	require.Len(t, backups, 2)
	oldest := backups[len(backups)-1]
	assert.Equal(t, mcp, oldest.Path)

	code, _, _ = runCLITest(t, "restore", oldest.ID)
	assert.Equal(t, exitConfirm, code)
	code, _, errOut = runCLITest(t, "restore", "--yes", oldest.ID)
	require.Equal(t, exitOK, code, errOut)
	code, _, _ = runCLITest(t, "restore", "--yes", "20200101-000000/nope")
	assert.Equal(t, exitError, code)
}

func readTestFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	require.NoError(t, err)
	return string(b)
}
