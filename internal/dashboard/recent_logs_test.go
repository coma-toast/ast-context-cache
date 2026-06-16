package dashboard

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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

func TestBuildRecentLogsMissingFile(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", filepath.Join(t.TempDir(), "missing.log"))
	lines, path, _ := buildRecentLogs(10)
	if path == "" {
		t.Fatal("empty path")
	}
	if len(lines) != 1 || lines[0].Level != "warn" {
		t.Fatalf("lines=%+v", lines)
	}
}
