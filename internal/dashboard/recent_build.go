package dashboard

import (
	"encoding/json"
	"path/filepath"
	"strconv"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const recentSelect = `SELECT timestamp, tool_name, result_chars, duration_ms, COALESCE(cpu_ms,0), project_path,
	COALESCE(error,''), COALESCE(arguments,''), COALESCE(tokens_saved,0), COALESCE(dedup_tokens_saved,0) FROM queries`

func buildRecentQueries(projectID string, limit int) (mcp, indexing []components.RecentQuery) {
	mcpLim, idxLim := 40, 25
	if limit > 0 {
		mcpLim = limit * 2 / 3
		idxLim = limit / 3
		if mcpLim < 10 {
			mcpLim = 10
		}
		if idxLim < 10 {
			idxLim = 10
		}
	}
	return queryRecent(projectID, mcpLim, excludeWatcherFromToolStats),
		queryRecent(projectID, idxLim, "tool_name = 'file_watcher'")
}

func queryRecent(projectID string, limit int, toolFilter string) []components.RecentQuery {
	if limit > 500 {
		limit = 500
	}
	where := toolFilter
	args := []any{}
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	args = append(args, limit)
	q := recentSelect + " WHERE " + where + " ORDER BY timestamp DESC LIMIT ?"
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []components.RecentQuery
	for rows.Next() {
		var ts, toolName, pp, errMsg, argsJSON string
		var saved, dedupSaved, rc int
		var dm, cpuMs float64
		if err := rows.Scan(&ts, &toolName, &rc, &dm, &cpuMs, &pp, &errMsg, &argsJSON, &saved, &dedupSaved); err != nil {
			continue
		}
		out = append(out, parseRecentRow(ts, toolName, pp, errMsg, argsJSON, saved, dedupSaved, dm, cpuMs))
	}
	return out
}

func parseRecentRow(ts, toolName, pp, errMsg, argsJSON string, saved, dedupSaved int, dm, cpuMs float64) components.RecentQuery {
	q := components.RecentQuery{
		ToolName:       toolName,
		Project:        pp,
		DurationMs:     dm,
		CpuMs:          cpuMs,
		Error:          errMsg,
		Saved:          saved,
		DedupTokensSaved: dedupSaved,
	}
	if t := parseQueryTime(ts); !t.IsZero() {
		q.Timestamp = formatRelativeTime(t)
		q.TimestampTitle = t.Format(time.RFC3339)
	} else {
		q.Timestamp = ts
		q.TimestampTitle = ts
	}
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(argsJSON), &parsed) == nil {
		if a, ok := parsed["arguments"].(map[string]interface{}); ok {
			parsed = a
		}
		if toolName == "file_watcher" {
			q.Event = stringVal(parsed, "event")
			if f := stringVal(parsed, "file"); f != "" {
				q.File = filepath.Base(f)
				q.FileTitle = f
			}
		} else {
			if queryStr, ok := parsed["query"].(string); ok {
				q.Query = queryStr
			}
			if m, ok := parsed["mode"].(string); ok && m != "" {
				q.Mode = m
			}
			if tb, ok := parsed["token_budget"].(float64); ok && tb > 0 {
				q.Budget = int(tb)
			}
		}
	}
	return q
}

func parseQueryTime(ts string) time.Time {
	t, err := time.Parse(time.RFC3339, ts)
	if err == nil {
		return t
	}
	t, err = time.Parse("2006-01-02T15:04:05Z07:00", ts)
	if err == nil {
		return t
	}
	t, _ = time.Parse("2006-01-02 15:04:05", ts)
	return t
}

func formatRelativeTime(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < 10*time.Second:
		return "just now"
	case d < time.Minute:
		return strconv.Itoa(int(d.Seconds())) + "s ago"
	case d < time.Hour:
		return strconv.Itoa(int(d.Minutes())) + "m ago"
	case d < 24*time.Hour:
		return strconv.Itoa(int(d.Hours())) + "h ago"
	case d < 7*24*time.Hour:
		return strconv.Itoa(int(d.Hours()/24)) + "d ago"
	default:
		return t.Format("Jan 2 15:04")
	}
}

func stringVal(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
