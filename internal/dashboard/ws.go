package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
	"github.com/gorilla/websocket"
)

// ─── WebSocket hub ────────────────────────────────────────────────────

type wsMsg struct {
	Type      string      `json:"type"`
	Timestamp string      `json:"timestamp"`
	Data      interface{} `json:"data"`
}

type wsClient struct {
	hub  *wsHub
	conn *websocket.Conn
	send chan []byte
}

type wsHub struct {
	clients    map[*wsClient]bool
	register   chan *wsClient
	unregister chan *wsClient
	broadcast  chan wsMsg
	mu         sync.Mutex
}

func newWSHub() *wsHub {
	return &wsHub{
		clients:    make(map[*wsClient]bool),
		register:   make(chan *wsClient),
		unregister: make(chan *wsClient),
		broadcast:  make(chan wsMsg, 512),
	}
}

func (h *wsHub) run() {
	for {
		select {
		case c := <-h.register:
			h.clients[c] = true
		case c := <-h.unregister:
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
			}
		case msg := <-h.broadcast:
			data, _ := json.Marshal(msg)
			h.mu.Lock()
			for c := range h.clients {
				select {
				case c.send <- data:
				default:
					close(c.send)
					delete(h.clients, c)
				}
			}
			h.mu.Unlock()
		}
	}
}

func (c *wsClient) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()
	for {
		if _, _, err := c.conn.ReadMessage(); err != nil {
			break
		}
	}
}

func (c *wsClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()
	for {
		select {
		case msg, ok := <-c.send:
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-ticker.C:
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, "Could not open WebSocket", http.StatusInternalServerError)
		return
	}
	c := &wsClient{hub: hub, conn: conn, send: make(chan []byte, 256)}
	c.hub.register <- c
	go c.writePump()
	go c.readPump()
}

// ─── Partial rendering helpers (render into string) ──────────────────

func renderIndexHealth() string {
	pid := ""
	h := components.IndexHealth{}
	if pid != "" {
		db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", pid).Scan(&h.TotalSymbols, &h.TotalFiles)
		db.DB.QueryRow("SELECT COUNT(*) FROM edges WHERE project_path = ?", pid).Scan(&h.TotalEdges)
	} else {
		db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&h.TotalSymbols, &h.TotalFiles)
		db.DB.QueryRow("SELECT COUNT(*) FROM edges").Scan(&h.TotalEdges)
	}
	h.TotalVectors = search.Cache.Count(pid)
	h.VectorMemMB = search.Cache.MemoryMB()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	h.MemoryMB = float64(memStats.Alloc) / (1024 * 1024)

	dbPath := db.GetDBPath()
	if fi, err := os.Stat(dbPath); err == nil {
		diskBytes := fi.Size()
		switch {
		case diskBytes >= 1024*1024*1024:
			h.DiskSize = fmt.Sprintf("%.2f GB", float64(diskBytes)/(1024*1024*1024))
		case diskBytes >= 1024*1024:
			h.DiskSize = fmt.Sprintf("%.1f MB", float64(diskBytes)/(1024*1024))
		case diskBytes >= 1024:
			h.DiskSize = fmt.Sprintf("%d KB", diskBytes/1024)
		default:
			h.DiskSize = fmt.Sprintf("%d B", diskBytes)
		}
	} else {
		h.DiskSize = "-"
	}

	ws := watcher.GetStatus()
	if watchers, ok := ws["watchers"].([]map[string]interface{}); ok {
		for _, w := range watchers {
			pp, _ := w["project_path"].(string)
			active, _ := w["active"].(bool)
			name := filepath.Base(pp)
			h.Watchers = append(h.Watchers, components.WatcherInfo{
				ProjectPath: pp,
				Name:        name,
				Active:      active,
			})
		}
	}
	q, activeEmb, workers, completed, throughput := embedqueue.Stats()
	h.EmbedQueued = q
	h.EmbedActive = int(activeEmb)
	h.EmbedWorkers = workers
	h.EmbedComplete = completed
	h.EmbedThroughput = throughput
	h.PinnedCount = db.PinnedProjectCount()
	var buf bytes.Buffer
	components.IndexHealthCards(h).Render(context.Background(), &buf)
	return buf.String()
}

func renderHealthBar() string {
	state, lastUse := mcp.EmbedderState()
	q, _, workers, _, throughput := embedqueue.Stats()
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	heapMB := float64(memStats.HeapAlloc) / (1024 * 1024)
	cacheHit := cache.GlobalCache.HitRatio()
	h := components.Health{
		EmbedderState:    state,
		EmbedderLast:    lastUse,
		QueueWorkers:    workers,
		QueueThroughput: throughput,
		QueueQueued:     q,
		CacheHitRatio:   cacheHit,
		HeapMB:          heapMB,
		Uptime:          time.Since(serverStartTime),
		Version:         "2.0.0",
	}
	var buf bytes.Buffer
	components.HealthBar(h).Render(context.Background(), &buf)
	return buf.String()
}

func renderStats() string {
	var s components.Stats
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	tokensSavedSum := "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN tokens_saved ELSE 0 END),0)"
	db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), "+tokensSavedSum+" FROM queries").
		Scan(&s.TotalQueries, &s.Sessions, &s.TotalChars, &s.AvgDurationMs, &s.TokensSaved)
	db.DB.QueryRow("SELECT COUNT(*), "+tokensSavedSum+" FROM queries WHERE timestamp >= ? AND timestamp < ?", todayStart, tomorrowStart).
		Scan(&s.TodayQueries, &s.TodayTokens)
	var buf bytes.Buffer
	components.StatsCards(s).Render(context.Background(), &buf)
	return buf.String()
}

func renderRecent() string {
	lim := 50
	rows, err := db.DB.Query("SELECT timestamp, tool_name, result_chars, duration_ms, project_path, COALESCE(error,''), COALESCE(arguments,''), COALESCE(tokens_saved,0), COALESCE(file_baseline_tokens,0) FROM queries ORDER BY timestamp DESC LIMIT ?", lim)
	if err != nil {
		var buf bytes.Buffer
		components.RecentTable(nil).Render(context.Background(), &buf)
		return buf.String()
	}
	defer rows.Close()
	var queries []components.RecentQuery
	for rows.Next() {
		var ts, toolName, pp, errMsg, argsJSON string
		var rc, saved, fileBaseline int
		var dm float64
		rows.Scan(&ts, &toolName, &rc, &dm, &pp, &errMsg, &argsJSON, &saved, &fileBaseline)
		q := components.RecentQuery{
			ToolName:   toolName,
			Project:    pp,
			DurationMs: dm,
			Error:      errMsg,
			Saved:      saved,
		}
		t, _ := time.Parse("2006-01-02T15:04:05Z07:00", ts)
		if t.IsZero() {
			t, _ = time.Parse("2006-01-02 15:04:05", ts)
		}
		if !t.IsZero() {
			q.Timestamp = t.Format("Jan 2 15:04:05")
		} else {
			q.Timestamp = ts
		}
		queries = append(queries, q)
	}
	var buf bytes.Buffer
	components.RecentTable(queries).Render(context.Background(), &buf)
	return buf.String()
}

func renderSymbolChart() string {
	rows, err := db.DB.Query("SELECT kind, COUNT(*) as count FROM symbols GROUP BY kind ORDER BY count DESC")
	if err != nil {
		var buf bytes.Buffer
		components.BarChart(nil, 0).Render(context.Background(), &buf)
		return buf.String()
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var kind string
		var count int
		rows.Scan(&kind, &count)
		items = append(items, components.BarItem{Label: kind, Value: count})
	}
	var buf bytes.Buffer
	components.BarChart(items, 0).Render(context.Background(), &buf)
	return buf.String()
}

func renderLanguageChart() string {
	q := `SELECT CASE
		WHEN file LIKE '%.py' THEN 'Python' WHEN file LIKE '%.go' THEN 'Go'
		WHEN file LIKE '%.js' THEN 'JavaScript' WHEN file LIKE '%.jsx' THEN 'JSX'
		WHEN file LIKE '%.ts' THEN 'TypeScript' WHEN file LIKE '%.tsx' THEN 'TSX'
		WHEN file LIKE '%.sh' THEN 'Bash' WHEN file LIKE '%.fish' THEN 'Fish'
		ELSE 'Other' END as language, COUNT(*) as symbols FROM symbols GROUP BY language ORDER BY symbols DESC`
	rows, err := db.DB.Query(q)
	if err != nil {
		var buf bytes.Buffer
		components.BarChart(nil, 3).Render(context.Background(), &buf)
		return buf.String()
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var lang string
		var symbols int
		rows.Scan(&lang, &symbols)
		items = append(items, components.BarItem{Label: lang, Value: symbols})
	}
	var buf bytes.Buffer
	components.BarChart(items, 3).Render(context.Background(), &buf)
	return buf.String()
}

func renderToolChart() string {
	rows, err := db.DB.Query("SELECT tool_name, COUNT(*) FROM queries GROUP BY tool_name ORDER BY COUNT(*) DESC")
	if err != nil {
		var buf bytes.Buffer
		components.BarChart(nil, 1).Render(context.Background(), &buf)
		return buf.String()
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var name string
		var count int
		rows.Scan(&name, &count)
		items = append(items, components.BarItem{Label: name, Value: count})
	}
	var buf bytes.Buffer
	components.BarChart(items, 1).Render(context.Background(), &buf)
	return buf.String()
}

func renderImportChart() string {
	rows, err := db.DB.Query("SELECT target, COUNT(*) as count FROM edges GROUP BY target ORDER BY count DESC LIMIT 20")
	if err != nil {
		var buf bytes.Buffer
		components.BarChart(nil, 5).Render(context.Background(), &buf)
		return buf.String()
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var target string
		var count int
		rows.Scan(&target, &count)
		items = append(items, components.BarItem{Label: target, Value: count})
	}
	var buf bytes.Buffer
	components.BarChart(items, 5).Render(context.Background(), &buf)
	return buf.String()
}

func renderSettings() string {
	settings := db.GetAllSettings()
	idleMinutes := 1
	if v, ok := settings["idle_unload_minutes"]; ok {
		if parsed, err := strconv.Atoi(v); err == nil {
			idleMinutes = parsed
		}
	}
	watcherIgn := settings["watcher_ignore_globs"]
	if watcherIgn == "" {
		watcherIgn = "[]"
	}
	indexLog := settings["index_log_files"] == "true"
	logRoots := settings["log_retention_roots"]
	if logRoots == "" {
		logRoots = "[]"
	}
	logRetentionMaxAge := 0
	if v, ok := settings["log_retention_max_age_days"]; ok && v != "" {
		logRetentionMaxAge, _ = strconv.Atoi(v)
	}
	logRetentionMaxMB := 0
	if v, ok := settings["log_retention_max_total_mib"]; ok && v != "" {
		logRetentionMaxMB, _ = strconv.Atoi(v)
	}
	logRetentionEn := settings["log_retention_enabled"] == "true"
	logDry := settings["log_retention_dry_run"] == "true"
	logLast := settings["log_retention_last_run"]
	projects := loadProjects("")
	configs, _ := db.GetAgentConfigs()
	var agents []components.AgentInfo
	for _, sa := range supportedAgents {
		a := components.AgentInfo{
			Type:        sa.Type,
			Name:        sa.Name,
			GlobalPath:  sa.GlobalPath,
			ProjectPath: sa.ProjectPath,
			Description: sa.Description,
		}
		for _, c := range configs {
			if c.AgentType == sa.Type {
				if c.IsGlobal {
					a.GlobalInstalled = true
				} else {
					a.ProjectInstalled = true
				}
			}
		}
		agents = append(agents, a)
	}
	var docSources []components.DocSource
	if sources, err := docs.ListSources(); err == nil {
		for _, s := range sources {
			updated := "Never"
			if s.LastUpdated != "" {
				if t, err := time.Parse("2006-01-02T15:04:05Z07:00", s.LastUpdated); err == nil {
					updated = t.Format("Jan 2, 2006 15:04")
				} else if t, err := time.Parse("2006-01-02 15:04:05", s.LastUpdated); err == nil {
					updated = t.Format("Jan 2, 2006 15:04")
				} else {
					updated = s.LastUpdated
				}
			}
			docSources = append(docSources, components.DocSource{
				ID:          s.ID,
				Name:        s.Name,
				Type:        s.Type,
				URL:         s.URL,
				Version:     s.Version,
				LastUpdated: updated,
			})
		}
	}
	data := components.SettingsData{
		IdleUnloadMinutes:       idleMinutes,
		WatcherIgnoreGlobs:     watcherIgn,
		IndexLogFiles:          indexLog,
		LogRetentionEnabled:    logRetentionEn,
		LogRetentionRoots:      logRoots,
		LogRetentionMaxAgeDays: logRetentionMaxAge,
		LogRetentionMaxTotalMB: logRetentionMaxMB,
		LogRetentionDryRun:     logDry,
		LogRetentionLastRun:    logLast,
		Projects:               projects,
		Agents:                 agents,
		DocSources:             docSources,
	}
	var buf bytes.Buffer
	components.Settings(data).Render(context.Background(), &buf)
	return buf.String()
}

func renderActivityChart() string {
	var buf bytes.Buffer
	buf.WriteString(`<div class="card"><div class="chart-container"><canvas id="activity-canvas"></canvas></div></div>`)
	return buf.String()
}

// ─── Background polling loops ─────────────────────────────────────────

func (h *wsHub) pollLoop(name string, interval time.Duration, renderFn func() string, target string) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	var last string
	for range ticker.C {
		html := renderFn()
		if html != last {
			last = html
			h.broadcast <- wsMsg{
				Type:      "partial",
				Timestamp: time.Now().Format(time.RFC3339),
				Data: map[string]string{
					"target": target,
					"html":   html,
				},
			}
		}
	}
}

// ─── Init hub and register routes ─────────────────────────────────────

var hub *wsHub

func init() {
	hub = newWSHub()
	go hub.run()
	go hub.pollLoop("index-health", 30*time.Second, renderIndexHealth, "#index-health")
	go hub.pollLoop("health-bar", 5*time.Second, renderHealthBar, "#health-bar")
	go hub.pollLoop("stats", 30*time.Second, renderStats, "#stats-cards")
	go hub.pollLoop("recent", 30*time.Second, renderRecent, "#recent-queries")
	go hub.pollLoop("symbol-chart", 30*time.Second, renderSymbolChart, "#symbol-chart")
	go hub.pollLoop("language-chart", 30*time.Second, renderLanguageChart, "#lang-chart")
	go hub.pollLoop("tool-chart", 30*time.Second, renderToolChart, "#tool-chart")
	go hub.pollLoop("import-chart", 30*time.Second, renderImportChart, "#import-chart")
	go hub.pollLoop("settings", 30*time.Second, renderSettings, "#settings-content")
}

func handleToastWS(w http.ResponseWriter, r *http.Request) {
	handleWS(w, r)
}

func broadcastToastWS(toolName, query, timeStr, savedText, durationMs, toolColor string) {
	if hub != nil {
		hub.broadcast <- wsMsg{
			Type:      "toast",
			Timestamp: timeStr,
			Data: map[string]string{
				"toolName":   toolName,
				"query":      query,
				"timeStr":    timeStr,
				"savedText":  savedText,
				"durationMs": durationMs,
				"toolColor":  toolColor,
			},
		}
	}
}