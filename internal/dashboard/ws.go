package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/version"
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
	go pushInitialDashboardSnapshot(c)
	go c.readPump()
}

// ─── Partial rendering helpers (render into string) ──────────────────

func renderIndexHealth() string {
	var buf bytes.Buffer
	components.IndexHealthCards(buildIndexHealth("")).Render(context.Background(), &buf)
	return buf.String()
}

func renderMemory() string {
	var buf bytes.Buffer
	components.MemoryPanel(buildMemory("", 1)).Render(context.Background(), &buf)
	return buf.String()
}

func renderHealthBar() string {
	state, lastUse := mcp.EmbedderState()
	embedErr := mcp.EmbedderError()
	eq := embedqueue.Snapshot()
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	heapMB := float64(memStats.HeapAlloc) / (1024 * 1024)
	cacheHit := cache.GlobalCache.HitRatio()
	h := components.Health{
		EmbedderState:    state,
		EmbedderError:    embedErr,
		EmbedderLast:    lastUse,
		QueueWorkers:    eq.Workers,
		QueueThroughput: eq.Throughput,
		QueueQueued:     eq.Queued,
		QueuePending:     eq.Pending,
		QueuePendingPeak: eq.PendingPeak,
		QueueInFlight:   eq.InFlight,
		QueueHighCap:    eq.HighCap,
		QueueLowCap:     eq.LowCap,
		CacheHitRatio:   cacheHit,
		HeapMB:          heapMB,
		CPUPercent:      sys.ProcessCPUPercent(),
		Uptime:          time.Since(serverStartTime),
		Version:         version.Version,
	}
	applyActiveEmbedderHealth(&h)
	var buf bytes.Buffer
	components.HealthBar(h).Render(context.Background(), &buf)
	return buf.String()
}

func renderStats() string {
	var s components.Stats
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	statsSel := "SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), " + tokensSavedSum + ", " + dedupTokensSum + ", " + savingsVsFilesSum + " FROM queries WHERE "
	where, args := statsQueriesWhere("")
	db.DB.QueryRow(statsSel+where, args...).
		Scan(&s.TotalQueries, &s.Sessions, &s.TotalChars, &s.AvgDurationMs, &s.TokensSaved, &s.DedupTokensSaved, &s.SavingsVsFiles)
	fillTodayStats("", todayStart, tomorrowStart, &s)
	fillVirtualContextStats(&s, "")
	var buf bytes.Buffer
	components.StatsCards(s).Render(context.Background(), &buf)
	return buf.String()
}

func renderRecent() string {
	mcp, indexing := buildRecentQueries("", 50)
	logs, logPath, logTrunc, logOpts := buildRecentLogsForDashboard()
	var buf bytes.Buffer
	components.RecentPanel(mcp, indexing, logs, logPath, logTrunc, logOpts).Render(context.Background(), &buf)
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
	var buf bytes.Buffer
	components.ToolPerformancePanel(queryToolStats("")).Render(context.Background(), &buf)
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
	watcherIgn := ignorepatterns.JSONForSettings(settings["watcher_ignore_globs"])
	projectExclude := projectmeta.ExcludeJSONForSettings(settings["project_exclude_paths"])
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
	queryRetentionEn := settings["query_retention_enabled"] != "false"
	queryRetentionMaxAge := 90
	if v, ok := settings["query_retention_max_age_days"]; ok && v != "" {
		queryRetentionMaxAge, _ = strconv.Atoi(v)
	}
	queryRetentionLast := settings["query_retention_last_run"]
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
	data := components.SettingsData{
		IdleUnloadMinutes:       idleMinutes,
		WatcherIgnoreGlobs:     watcherIgn,
		ProjectExcludePaths:    projectExclude,
		IndexLogFiles:          indexLog,
		LogRetentionEnabled:    logRetentionEn,
		LogRetentionRoots:      logRoots,
		LogRetentionMaxAgeDays: logRetentionMaxAge,
		LogRetentionMaxTotalMB: logRetentionMaxMB,
		LogRetentionDryRun:     logDry,
		LogRetentionLastRun:    logLast,
		QueryRetentionEnabled:   queryRetentionEn,
		QueryRetentionMaxAgeDays: queryRetentionMaxAge,
		QueryRetentionLastRun:   queryRetentionLast,
		Projects:               projects,
		Agents:                 agents,
		EmbedWorkerMax:         embedqueue.MaxWorkers(),
	}
	PopulateEmbedSettings(settings, &data)
	populateContextSettings(settings, &data)
	applyActiveEmbedderSettings(&data)
	loadEmbedModels(&data)
	var buf bytes.Buffer
	components.Settings(data).Render(context.Background(), &buf)
	return buf.String()
}

func renderActivityChart() string {
	var buf bytes.Buffer
	buf.WriteString(`<div class="card"><div class="chart-container"><canvas id="activity-canvas"></canvas></div></div>`)
	return buf.String()
}

type dashboardPartial struct {
	name   string
	target string
	render func() string
}

// Panels refreshed via internal/realtime.Notify (no server polling).
var dashboardPartials = []dashboardPartial{
	{"index-health", "#index-health", renderIndexHealth},
	{"memory", "#memory-panel", renderMemory},
	{"health-bar", "#health-bar", renderHealthBar},
	{"stats", "#stats-cards", renderStats},
	{"recent", "#recent-queries", renderRecent},
	{"symbol-chart", "#symbol-chart", renderSymbolChart},
	{"language-chart", "#lang-chart", renderLanguageChart},
	{"tool-chart", "#tool-chart", renderToolChart},
	{"import-chart", "#import-chart", renderImportChart},
	{"settings", "#settings-content", renderSettings},
}

func pushInitialDashboardSnapshot(c *wsClient) {
	for _, p := range dashboardPartials {
		html := p.render()
		data, err := json.Marshal(wsMsg{
			Type:      "partial",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]string{
				"target": p.target,
				"html":   html,
			},
		})
		if err != nil {
			continue
		}
		select {
		case c.send <- data:
		default:
			return
		}
	}
}

// ─── Init hub and register routes ─────────────────────────────────────

var hub *wsHub

func init() {
	hub = newWSHub()
	go hub.run()
	initRealtimeBridge()
	initQueryLogBridge()
	initLogNotifyBridge()
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