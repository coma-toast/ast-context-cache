package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

func handleSSE(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	var lastTimestamp string

	// Get initial timestamp
	row := db.DB.QueryRow("SELECT timestamp FROM queries ORDER BY timestamp DESC LIMIT 1")
	row.Scan(&lastTimestamp)

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			rows, err := db.DB.Query("SELECT timestamp, tool_name, COALESCE(arguments,''), COALESCE(tokens_saved,0), duration_ms FROM queries WHERE timestamp > ? ORDER BY timestamp ASC", lastTimestamp)
			if err != nil {
				continue
			}
			for rows.Next() {
				var ts, toolName, argsJSON string
				var saved int
				var durationMs float64
				rows.Scan(&ts, &toolName, &argsJSON, &saved, &durationMs)
				lastTimestamp = ts

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

				t, _ := time.Parse("2006-01-02T15:04:05Z07:00", ts)
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

				toast := components.ToastData{
					ToolName:   toolName,
					Query:      query,
					TimeStr:    timeStr,
					SavedText:  savedText,
					DurationMs: durText,
					ToolColor:  components.GetToolColor(toolName),
				}

				var buf bytes.Buffer
				components.Toast(toast).Render(context.Background(), &buf)

				fmt.Fprintf(w, "event: toast\ndata: %s\n\n", buf.String())
				flusher.Flush()
			}
			rows.Close()
		}
	}
}
