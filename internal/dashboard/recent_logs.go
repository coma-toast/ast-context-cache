package dashboard

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const legacyServerLogPath = "/tmp/ast-mcp.log"

func serverLogPath() string {
	return db.ResolveServerLogPath()
}

func logViewOpts() components.LogViewOpts {
	opts := components.LogViewOpts{TailLines: 200, MaxLineChars: 500}
	if v := strings.TrimSpace(db.GetSetting("dashboard_log_tail_lines", "200")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			opts.TailLines = n
		}
	}
	if v := strings.TrimSpace(db.GetSetting("dashboard_log_line_chars", "500")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			opts.MaxLineChars = n
		}
	}
	if opts.TailLines < 50 {
		opts.TailLines = 50
	}
	if opts.TailLines > 500 {
		opts.TailLines = 500
	}
	if opts.MaxLineChars < 80 {
		opts.MaxLineChars = 80
	}
	if opts.MaxLineChars > 8000 {
		opts.MaxLineChars = 8000
	}
	return opts
}

func buildRecentLogsForDashboard() (lines []components.RecentLogLine, path string, fileTruncated bool, opts components.LogViewOpts) {
	opts = logViewOpts()
	lines, path, fileTruncated = buildRecentLogs(opts.TailLines)
	for i := range lines {
		lines[i] = truncateLogDisplay(lines[i], opts.MaxLineChars)
	}
	return lines, path, fileTruncated, opts
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
			msg = "Log file not found at " + path + " — use ast-mcp start, mcp-local start, or set AST_MCP_LOG_PATH"
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

func truncateLogDisplay(line components.RecentLogLine, maxChars int) components.RecentLogLine {
	if maxChars <= 0 || len(line.Message) <= maxChars {
		return line
	}
	line.MsgTruncated = true
	line.Message = line.Message[:maxChars] + "…"
	return line
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
	case strings.Contains(lower, "error") ||
		strings.Contains(lower, "fatal") ||
		strings.Contains(lower, "timeout") ||
		strings.Contains(lower, "deadline exceeded") ||
		strings.Contains(lower, " failed") ||
		strings.HasSuffix(lower, " failed"):
		line.Level = "error"
	case strings.Contains(lower, "warn") ||
		strings.Contains(lower, "throttl") ||
		strings.Contains(lower, "locked") ||
		strings.Contains(lower, "busy=1"):
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
