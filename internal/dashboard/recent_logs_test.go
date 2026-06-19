package dashboard

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestParseLogLine(t *testing.T) {
	line := parseLogLine("2026/06/16 12:34:56 embed queue depth=3")
	if line.Timestamp != "2026/06/16 12:34:56" {
		t.Fatalf("timestamp=%q", line.Timestamp)
	}
	if line.Message != "embed queue depth=3" {
		t.Fatalf("message=%q", line.Message)
	}
	if line.Level != "info" {
		t.Fatalf("level=%q", line.Level)
	}
	errLine := parseLogLine("2026/06/16 12:34:56 ERROR: connection failed")
	if errLine.Level != "error" {
		t.Fatalf("error level=%q", errLine.Level)
	}
	timeoutLine := parseLogLine("2026/06/19 14:15:17 embed: context deadline exceeded")
	if timeoutLine.Level != "error" {
		t.Fatalf("timeout level=%q", timeoutLine.Level)
	}
	warnLine := parseLogLine("2026/06/19 14:13:47 embed queue: throttled workers 10 -> 4")
	if warnLine.Level != "warn" {
		t.Fatalf("warn level=%q", warnLine.Level)
	}
}

func TestTailFileLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")
	content := strings.Join([]string{"line1", "line2", "line3", "line4", "line5"}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, _, err := tailFileLines(path, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 3 {
		t.Fatalf("len=%d", len(lines))
	}
	if lines[0] != "line3" || lines[2] != "line5" {
		t.Fatalf("lines=%v", lines)
	}
}

func TestServerLogPathDefault(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	want := filepath.Join(home, ".astcache", "ast-mcp.log")
	if got := serverLogPath(); got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestServerLogPathMcpLocalNewer(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	defaultPath := db.DefaultLogPath()
	mcpPath := db.McpLocalLogPath()
	os.MkdirAll(filepath.Dir(defaultPath), 0755)
	os.MkdirAll(filepath.Dir(mcpPath), 0755)
	os.WriteFile(defaultPath, []byte("a\n"), 0o644)
	time.Sleep(15 * time.Millisecond)
	os.WriteFile(mcpPath, []byte("b\n"), 0o644)
	if got := serverLogPath(); got != mcpPath {
		t.Fatalf("got %q want %q", got, mcpPath)
	}
}

func TestServerLogPathLegacyFallback(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.WriteFile(legacyServerLogPath, []byte("legacy\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(legacyServerLogPath) })
	if got := serverLogPath(); got != legacyServerLogPath {
		t.Fatalf("got %q want legacy %q", got, legacyServerLogPath)
	}
}

func TestBuildRecentLogsMissingFile(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", filepath.Join(t.TempDir(), "missing.log"))
	lines, path, _ := buildRecentLogs(10)
	if path == "" {
		t.Fatal("empty path")
	}
	if len(lines) != 1 || lines[0].Level != "warn" {
		t.Fatalf("lines=%+v", lines)
	}
	if !strings.Contains(lines[0].Message, "Log file not found") {
		t.Fatalf("message=%q", lines[0].Message)
	}
}

func TestDefaultLogPathMatchesDB(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	want := filepath.Join(home, ".astcache", "ast-mcp.log")
	if got := db.DefaultLogPath(); got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
