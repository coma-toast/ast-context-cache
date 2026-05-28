package dashboard

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

func initQueryLogBridge() {
	db.AfterQueryLogFlush = onQueryLogFlush
}

func onQueryLogFlush(rows []db.QueryLogSnapshot) {
	for _, r := range rows {
		notifyQueryLogged(r.ToolName, r.ArgsJSON, r.Timestamp, r.TokensSaved, r.DurationMs, r.CpuMs)
	}
	if len(rows) > 0 {
		realtime.Notify(realtime.QueryLogged)
	}
}

func notifyQueryLogged(toolName, argsJSON, ts string, saved int, durationMs, cpuMs float64) {
	query := toolName
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(argsJSON), &parsed) == nil {
		if a, ok := parsed["arguments"].(map[string]interface{}); ok {
			parsed = a
		}
		if q, ok := parsed["query"].(string); ok && q != "" {
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
	broadcastToastWS(toolName, query, timeStr, savedText, durText, "inherit")
}
