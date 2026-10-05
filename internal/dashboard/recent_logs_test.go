package dashboard

import (
	"bytes"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/logging"
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

func TestParseSlogLogLine(t *testing.T) {
	tests := map[string]struct {
		raw       string
		timestamp string
		level     string
		message   string
	}{
		"text": {
			raw:       `time=2026-10-05T14:00:00.000-05:00 level=INFO msg="Wrote rows" tag=db rows=3`,
			timestamp: "2026-10-05T14:00:00.000-05:00",
			level:     "info",
			message:   "Wrote rows tag=db rows=3",
		},
		"text warn keeps quoted attrs": {
			raw:       `time=2026-10-05T14:00:00.000-05:00 level=WARN msg="Failed to flush" tag=db error="disk full: no space" path="/tmp/a b.db"`,
			timestamp: "2026-10-05T14:00:00.000-05:00",
			level:     "warn",
			message:   `Failed to flush tag=db error="disk full: no space" path="/tmp/a b.db"`,
		},
		"text info mentioning error is not promoted": {
			raw:       `time=2026-10-05T14:00:00.000-05:00 level=INFO msg="Cleared error count" tag=embed`,
			timestamp: "2026-10-05T14:00:00.000-05:00",
			level:     "info",
			message:   "Cleared error count tag=embed",
		},
		"text level offset": {
			raw:     `time=2026-10-05T14:00:00.000-05:00 level=ERROR+2 msg=Crashed`,
			level:   "error",
			message: "Crashed",
		},
		"text debug": {
			raw:     `time=2026-10-05T14:00:00.000-05:00 level=DEBUG msg=Polled tag=watcher`,
			level:   "debug",
			message: "Polled tag=watcher",
		},
		"json": {
			raw:       `{"time":"2026-10-05T14:00:00.000-05:00","level":"ERROR","msg":"Failed to open db","tag":"db","error":"locked","attempt":2,"meta":{"a":1}}`,
			timestamp: "2026-10-05T14:00:00.000-05:00",
			level:     "error",
			message:   `Failed to open db tag=db error=locked attempt=2 meta={"a":1}`,
		},
		"json quotes values with spaces": {
			raw:     `{"time":"t","level":"WARN","msg":"Skipped","path":"/a b"}`,
			level:   "warn",
			message: `Skipped path="/a b"`,
		},
		"json without slog keys falls back to legacy": {
			raw:     `{"foo":"bar"}`,
			level:   "info",
			message: `{"foo":"bar"}`,
		},
		"malformed text falls back to legacy": {
			raw:     `time=broken "unterminated level=ERROR`,
			level:   "error",
			message: `time=broken "unterminated level=ERROR`,
		},
		"legacy stdlib line": {
			raw:       "2026/06/16 12:34:56 embed queue depth=3",
			timestamp: "2026/06/16 12:34:56",
			level:     "info",
			message:   "embed queue depth=3",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			line := parseLogLine(tc.raw)
			assert.Equal(t, tc.raw, line.Raw)
			assert.Equal(t, tc.level, line.Level)
			assert.Equal(t, tc.message, line.Message)
			if tc.timestamp != "" {
				assert.Equal(t, tc.timestamp, line.Timestamp)
			}
		})
	}
}

func TestParseLogLineFromLoggingHandler(t *testing.T) {
	for _, format := range []string{logging.FormatText, logging.FormatJSON} {
		t.Run(format, func(t *testing.T) {
			var buf bytes.Buffer
			logger := slog.New(logging.NewHandler(&buf, format, slog.LevelDebug)).With("tag", "db")
			logger.Warn("Failed to flush", "path", "/tmp/a b.db", "error", errors.New("disk full"))
			line := parseLogLine(strings.TrimSuffix(buf.String(), "\n"))
			assert.Equal(t, "warn", line.Level)
			assert.NotEmpty(t, line.Timestamp)
			assert.Equal(t, `Failed to flush tag=db path="/tmp/a b.db" error="disk full"`, line.Message)
		})
	}
}

func TestBuildRecentLogsMixedFormats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ast-mcp.log")
	t.Setenv("AST_MCP_LOG_PATH", path)
	content := strings.Join([]string{
		"2026/06/16 12:34:56 legacy line failed",
		`time=2026-10-05T14:00:00.000-05:00 level=INFO msg="Wrote rows" tag=db rows=3`,
		`{"time":"2026-10-05T14:00:01.000-05:00","level":"WARN","msg":"Throttled","tag":"embed"}`,
	}, "\n") + "\n"
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	lines, _, _ := buildRecentLogs(10)
	require.Len(t, lines, 3)
	assert.Equal(t, []string{"error", "info", "warn"}, []string{lines[0].Level, lines[1].Level, lines[2].Level})
	assert.Equal(t, "Wrote rows tag=db rows=3", lines[1].Message)
	assert.Equal(t, "Throttled tag=embed", lines[2].Message)
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
	os.MkdirAll(filepath.Dir(want), 0755)
	os.WriteFile(want, []byte("test\n"), 0o644)
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

func TestTruncateLogDisplay(t *testing.T) {
	line := components.RecentLogLine{Message: strings.Repeat("x", 120), Raw: strings.Repeat("x", 120)}
	out := truncateLogDisplay(line, 80)
	if !out.MsgTruncated {
		t.Fatal("expected truncated")
	}
	if !strings.HasSuffix(out.Message, "…") || len([]rune(out.Message)) != 81 {
		t.Fatalf("message=%q", out.Message)
	}
}

func TestLogViewOptsDefaults(t *testing.T) {
	dbtest.Init(t)
	_ = db.SetSetting("dashboard_log_tail_lines", "")
	_ = db.SetSetting("dashboard_log_line_chars", "")
	opts := logViewOpts()
	if opts.TailLines != 200 || opts.MaxLineChars != 500 {
		t.Fatalf("opts=%+v", opts)
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
