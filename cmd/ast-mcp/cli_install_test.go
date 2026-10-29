package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

// The CLI opens and closes the global db pools on every call, so these tests don't run in parallel.

func cliHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("DB_PATH", "")
	t.Setenv("AST_MCP_PORT", "")
	setDesktopRunning(t, false)
	return home
}

// setDesktopRunning stubs the Claude Desktop process check, so no test depends on whether the
// developer's own Claude Desktop is open.
func setDesktopRunning(t *testing.T, running bool) {
	t.Helper()
	claudeDesktopRunning = func() bool { return running }
	t.Cleanup(func() { claudeDesktopRunning = nil })
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

// mcp-local runs one install per target: the pre-4.0 warnings print on the first run only, while
// verify keeps showing them with a pointer to dismiss them.
func TestCLILegacyWarningsPrintOnce(t *testing.T) {
	cliHome(t)
	require.NoError(t, db.InitUsage())
	_, err := db.DB.Exec("INSERT INTO agent_configs (agent_type, install_path, is_global, instructions_hash) VALUES ('cursor', '~/.cursor/mcp.json', 1, 'h')")
	db.Close()
	require.NoError(t, err)
	legacy := func(out string) int {
		var res cliOutput
		require.NoError(t, json.Unmarshal([]byte(out), &res))
		n := 0
		for _, w := range res.Warnings {
			if strings.Contains(w, "pre-4.0") {
				n++
			}
		}
		return n
	}
	_, out, _ := runCLITest(t, "install", "--target", "cursor", "--dry-run", "--json")
	assert.Equal(t, 1, legacy(out), "first run shows them")
	for _, target := range []string{"cursor", "codex", "opencode"} {
		_, out, _ = runCLITest(t, "install", "--target", target, "--dry-run", "--json")
		assert.Zero(t, legacy(out), target)
	}
	_, out, _ = runCLITest(t, "verify", "--target", "cursor")
	assert.Contains(t, out, "pre-4.0 install replaced ~/.cursor/mcp.json")
	assert.Contains(t, out, "Dismiss the pre-4.0 install warnings in the dashboard")
}

// Applying while Claude Desktop runs waits and re-verifies: an entry the app drops in the
// meantime is reported as missing, with exit 3.
func TestCLIRecheckReportsDroppedDesktopEntry(t *testing.T) {
	home := cliHome(t)
	setDesktopRunning(t, true)
	path := filepath.Join(home, ".config", "Claude", "claude_desktop_config.json")
	if runtime.GOOS == "darwin" {
		path = filepath.Join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json")
	}
	const config = "{\n  \"mcpServers\": {}\n}\n"
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(config), 0o644))
	var waited time.Duration
	recheckSleep = func(d time.Duration) {
		waited = d
		// Claude Desktop writes back the config it holds in memory.
		require.NoError(t, os.WriteFile(path, []byte(config), 0o644))
	}
	t.Cleanup(func() { recheckSleep = time.Sleep })

	code, out, errOut := runCLITest(t, "install", "--target", "claude_desktop", "--component", "mcp", "--yes", "--json")
	assert.Equal(t, exitConflict, code, errOut)
	assert.Equal(t, 10*time.Second, waited)
	var res cliOutput
	require.NoError(t, json.Unmarshal([]byte(out), &res))
	require.NotEmpty(t, res.Status)
	assert.Equal(t, "mcp", res.Status[0].Component)
	assert.Equal(t, "missing", res.Status[0].Status)
	joined := strings.Join(res.Warnings, "\n")
	assert.Contains(t, joined, "Claude Desktop is running")
	assert.Contains(t, joined, "is now missing (was installed)")

	waited = 0
	code, out, errOut = runCLITest(t, "install", "--target", "claude_desktop", "--component", "mcp", "--yes", "--recheck", "0")
	assert.Equal(t, exitOK, code, errOut)
	assert.Zero(t, waited, "--recheck 0 skips the wait")
	assert.Contains(t, out, "quit Claude Desktop, then apply")
}
