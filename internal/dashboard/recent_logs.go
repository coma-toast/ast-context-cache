package dashboard

import (
	"encoding/json"
	"log/slog"
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

// parseLogLine turns one server log line into a display row. It understands slog JSON
// (AST_LOG_FORMAT=json) and slog text (the default) records, and falls back to a keyword
// heuristic for legacy stdlib-log lines and anything else (panics, third-party output).
func parseLogLine(raw string) components.RecentLogLine {
	trimmed := strings.TrimSpace(raw)
	if strings.HasPrefix(trimmed, "{") {
		if line, ok := parseJSONLogLine(raw, trimmed); ok {
			return line
		}
	}
	if strings.HasPrefix(trimmed, slog.TimeKey+"=") || strings.HasPrefix(trimmed, slog.LevelKey+"=") {
		if line, ok := parseTextLogLine(raw, trimmed); ok {
			return line
		}
	}
	return parseLegacyLogLine(raw)
}

// parseJSONLogLine reads a slog JSON record, keeping attribute order so the displayed
// message lists attributes the way they were logged.
func parseJSONLogLine(raw, s string) (components.RecentLogLine, bool) {
	dec := json.NewDecoder(strings.NewReader(s))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return components.RecentLogLine{}, false
	}
	line := components.RecentLogLine{Raw: raw, Level: "info"}
	var msg string
	var attrs []string
	recognized := false
	for dec.More() {
		tok, err := dec.Token()
		key, ok := tok.(string)
		if err != nil || !ok {
			return components.RecentLogLine{}, false
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return components.RecentLogLine{}, false
		}
		var str string
		isStr := json.Unmarshal(val, &str) == nil
		switch {
		case key == slog.TimeKey && isStr:
			line.Timestamp = str
		case key == slog.LevelKey && isStr:
			line.Level, recognized = normalizeLogLevel(str), true
		case key == slog.MessageKey && isStr:
			msg, recognized = str, true
		case isStr:
			attrs = append(attrs, key+"="+quoteLogValue(str))
		default:
			attrs = append(attrs, key+"="+string(val))
		}
	}
	if !recognized {
		return components.RecentLogLine{}, false
	}
	line.Message = joinLogMessage(msg, attrs)
	return line, true
}

// parseTextLogLine reads a slog text record (key=value pairs, Go-quoted values). Attributes
// other than time/level/msg are kept in their original key=value form for display.
func parseTextLogLine(raw, s string) (components.RecentLogLine, bool) {
	line := components.RecentLogLine{Raw: raw, Level: "info"}
	var msg string
	var attrs []string
	recognized := false
	for s = strings.TrimLeft(s, " "); s != ""; s = strings.TrimLeft(s, " ") {
		eq := strings.IndexByte(s, '=')
		if eq <= 0 || strings.ContainsAny(s[:eq], " \"") {
			return components.RecentLogLine{}, false
		}
		key, rest := s[:eq], s[eq+1:]
		val, tokenLen := rest, len(rest)
		if strings.HasPrefix(rest, `"`) {
			quoted, err := strconv.QuotedPrefix(rest)
			if err != nil {
				return components.RecentLogLine{}, false
			}
			val, _ = strconv.Unquote(quoted)
			tokenLen = len(quoted)
		} else if sp := strings.IndexByte(rest, ' '); sp >= 0 {
			val, tokenLen = rest[:sp], sp
		}
		switch key {
		case slog.TimeKey:
			line.Timestamp = val
		case slog.LevelKey:
			line.Level, recognized = normalizeLogLevel(val), true
		case slog.MessageKey:
			msg, recognized = val, true
		default:
			attrs = append(attrs, key+"="+rest[:tokenLen])
		}
		s = rest[tokenLen:]
	}
	if !recognized {
		return components.RecentLogLine{}, false
	}
	line.Message = joinLogMessage(msg, attrs)
	return line, true
}

// parseLegacyLogLine guesses the level of a stdlib-log ("2006/01/02 15:04:05 msg") or
// unstructured line from keywords, since such lines carry no level.
func parseLegacyLogLine(raw string) components.RecentLogLine {
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

// normalizeLogLevel maps slog level names, including offsets like "WARN+2", to the
// dashboard's lowercase levels.
func normalizeLogLevel(level string) string {
	switch upper := strings.ToUpper(strings.TrimSpace(level)); {
	case strings.HasPrefix(upper, "ERROR"):
		return "error"
	case strings.HasPrefix(upper, "WARN"):
		return "warn"
	case strings.HasPrefix(upper, "DEBUG"):
		return "debug"
	default:
		return "info"
	}
}

func quoteLogValue(v string) string {
	if v == "" || strings.ContainsAny(v, " \"=\t\n") {
		return strconv.Quote(v)
	}
	return v
}

func joinLogMessage(msg string, attrs []string) string {
	if len(attrs) == 0 {
		return msg
	}
	if msg == "" {
		return strings.Join(attrs, " ")
	}
	return msg + " " + strings.Join(attrs, " ")
}
