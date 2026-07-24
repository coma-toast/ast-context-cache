package dashboard

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

func initQueryLogBridge() {
	db.AfterQueryLogFlush = onQueryLogFlush
}

func onQueryLogFlush(rows []db.QueryLogSnapshot) {
	observeQueryLogMetrics(rows)
	for _, r := range rows {
		notifyQueryLogged(r.ToolName, r.ArgsJSON, r.Timestamp, r.ProjectPath, r.TokensSaved, r.DurationMs, r.CpuMs)
	}
	if len(rows) > 0 {
		realtime.Notify(realtime.QueryLogged)
	}
}

func notifyQueryLogged(toolName, argsJSON, ts, projectPath string, saved int, durationMs, cpuMs float64) {
	query := toolName
	displayName := toolName
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(argsJSON), &parsed) == nil {
		if a, ok := parsed["arguments"].(map[string]interface{}); ok {
			parsed = a
		}
		if toolName == "file_watcher" {
			query = formatWatcherToastQuery(parsed)
			displayName = "Indexing"
			if projectPath != "" {
				displayName += " · " + filepath.Base(projectPath)
			}
		} else if q, ok := parsed["query"].(string); ok && q != "" {
			if len(q) > 30 {
				query = q[:27] + "..."
			} else {
				query = q
			}
		}
	}
	t, _ := time.Parse(time.RFC3339, ts)
	if t.IsZero() {
		t, _ = time.Parse("2006-01-02 15:04:05", ts)
	}
	timeStr := t.Format("15:04:05")
	savedText := ""
	if saved > 0 {
		savedText = fmt.Sprintf("+%d tok", saved)
	}
	durText := ""
	if durationMs > 0 {
		durText = fmt.Sprintf("%.0fms", durationMs)
	}
	if cpuMs > 0 {
		cpuPart := fmt.Sprintf("%.0fms cpu", cpuMs)
		if durText != "" {
			durText += " · " + cpuPart
		} else {
			durText = cpuPart
		}
	}
	toolColor := "inherit"
	if toolName == "file_watcher" {
		toolColor = "#3fb950"
	}
	broadcastToastWS(displayName, query, timeStr, savedText, durText, toolColor)
}

func formatWatcherToastQuery(parsed map[string]interface{}) string {
	event := toastStringVal(parsed, "event")
	file := toastStringVal(parsed, "file")
	if file != "" {
		file = filepath.Base(file)
	}
	parts := make([]string, 0, 2)
	if event != "" {
		parts = append(parts, event)
	}
	if file != "" {
		parts = append(parts, file)
	}
	if len(parts) == 0 {
		return "index update"
	}
	return strings.Join(parts, " · ")
}

func toastStringVal(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
