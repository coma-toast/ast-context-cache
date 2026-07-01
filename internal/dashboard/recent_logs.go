package dashboard

import (
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const legacyServerLogPath = "/tmp/ast-mcp.log"

func serverLogPath() string {
	return db.ResolveServerLogPath()
}

func logViewOpts() components.LogViewOpts {
	return logViewOptsFast()
}

func buildRecentLogsForDashboard() (lines []components.RecentLogLine, path string, fileTruncated bool, opts components.LogViewOpts) {
	opts = logViewOptsFast()
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
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	size := fi.Size()
	if size == 0 {
		return nil, false, nil
	}
	const chunkSize = 64 * 1024
	var buf []byte
	truncated := false
	offset := size
	for offset > 0 {
		readAt := offset
		readSize := int64(chunkSize)
		if readSize > readAt {
			readSize = readAt
		}
		offset -= readSize
		chunk := make([]byte, readSize)
		if _, err := f.ReadAt(chunk, offset); err != nil {
			return nil, false, err
		}
		buf = append(chunk, buf...)
		lineCount := strings.Count(string(buf), "\n")
		if lineCount > maxLines+1 {
			truncated = offset > 0
			break
		}
		if offset == 0 {
			break
		}
		truncated = true
	}
	text := string(buf)
	if text == "" {
		return nil, false, nil
	}
	lines := strings.Split(strings.TrimSuffix(text, "\n"), "\n")
	if len(lines) > maxLines {
		truncated = true
		lines = lines[len(lines)-maxLines:]
	}
	return lines, truncated, nil
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
