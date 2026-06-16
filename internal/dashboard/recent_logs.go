package dashboard

import (
	"bufio"
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const legacyServerLogPath = "/tmp/ast-mcp.log"

func serverLogPath() string {
	if p := strings.TrimSpace(os.Getenv("AST_MCP_LOG_PATH")); p != "" {
		return p
	}
	p := db.DefaultLogPath()
	if _, err := os.Stat(p); err == nil {
		return p
	}
	if _, err := os.Stat(legacyServerLogPath); err == nil {
		return legacyServerLogPath
	}
	return p
}

func buildRecentLogs(maxLines int) (lines []components.RecentLogLine, path string, truncated bool) {
	path = serverLogPath()
	if maxLines <= 0 {
		maxLines = 200
	}
	if maxLines > 500 {
		maxLines = 500
	}
	raw, trunc, err := tailFileLines(path, maxLines)
	if err != nil {
		msg := err.Error()
		if os.IsNotExist(err) {
			msg = "Log file not found at " + path + " — use ast-mcp start (logs to ~/.astcache/ast-mcp.log) or set AST_MCP_LOG_PATH"
		}
		return []components.RecentLogLine{{
			Level:   "warn",
			Message: msg,
			Raw:     msg,
		}}, path, false
	}
	for _, line := range raw {
		lines = append(lines, parseLogLine(line))
	}
	return lines, path, trunc
}

func tailFileLines(path string, maxLines int) ([]string, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, err
		}
		return nil, false, err
	}
	defer f.Close()
	var ring []string
	truncated := false
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		if len(ring) >= maxLines {
			truncated = true
			ring = ring[1:]
		}
		ring = append(ring, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, false, err
	}
	return ring, truncated, nil
}

func parseLogLine(raw string) components.RecentLogLine {
	line := components.RecentLogLine{Raw: raw, Message: raw}
	lower := strings.ToLower(raw)
	switch {
	case strings.Contains(lower, "error") || strings.Contains(lower, "fatal"):
		line.Level = "error"
	case strings.Contains(lower, "warn"):
		line.Level = "warn"
	default:
		line.Level = "info"
	}
	if len(raw) >= 20 && raw[4] == '/' && raw[7] == '/' {
		line.Timestamp = strings.TrimSpace(raw[:19])
		line.Message = strings.TrimSpace(raw[20:])
	}
	return line
}
