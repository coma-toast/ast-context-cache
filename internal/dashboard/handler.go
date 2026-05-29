package dashboard

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

var serverStartTime = time.Now()

func handleDashboardPage(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	projects := loadProjects("")
	h := components.HealthInfo{
		Version:    version.Version,
		StartTime: serverStartTime,
	}
	components.PageTemplate(projects, h).Render(r.Context(), w)
}

func handleStatsPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	s := components.Stats{}
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	statsSel := "SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), " + tokensSavedSum + ", " + dedupTokensSum + ", " + savingsVsFilesSum + " FROM queries WHERE "
	where, args := statsQueriesWhere(pid)
	db.DB.QueryRow(statsSel+where, args...).
		Scan(&s.TotalQueries, &s.Sessions, &s.TotalChars, &s.AvgDurationMs, &s.TokensSaved, &s.DedupTokensSaved, &s.SavingsVsFiles)
	fillTodayStats(pid, todayStart, tomorrowStart, &s)
	components.StatsCards(s).Render(r.Context(), w)
}

func handleActivityPartial(w http.ResponseWriter, r *http.Request) {
	activity := embedqueue.RecentActivity()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"activity": activity})
}

func handleHealthPartial(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")

	state, lastUse := mcp.EmbedderState()

	eq := embedqueue.Snapshot()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	heapMB := float64(memStats.HeapAlloc) / (1024 * 1024)

	cacheHit := cache.GlobalCache.HitRatio()

	h := components.Health{
		EmbedderState:    state,
		EmbedderLast:    lastUse,
		QueueWorkers:   eq.Workers,
		QueueThroughput: eq.Throughput,
		QueueQueued:     eq.Queued,
		QueueInFlight:   eq.InFlight,
		QueueHighCap:    eq.HighCap,
		QueueLowCap:     eq.LowCap,
		CacheHitRatio: cacheHit,
		HeapMB:         heapMB,
		CPUPercent:     sys.ProcessCPUPercent(),
		Uptime:        time.Since(serverStartTime),
		Version:       version.Version,
	}
	applyActiveEmbedderHealth(&h)
	components.HealthBar(h).Render(r.Context(), w)
}

func handleIndexHealthPartial(w http.ResponseWriter, r *http.Request) {
	components.IndexHealthCards(buildIndexHealth(r.URL.Query().Get("project_id"))).Render(r.Context(), w)
}

func handleActivityDataPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	interval := r.URL.Query().Get("interval")
	if interval == "" {
		interval = "daily"
	}
	days := 30
	if d := r.URL.Query().Get("days"); d != "" {
		fmt.Sscanf(d, "%d", &days)
	}
	if days < 1 {
		days = 1
	} else if days > 365 {
		days = 365
	}
	var format string
	if interval == "hourly" {
		format = "%Y-%m-%dT%H:00"
	} else {
		format = "%Y-%m-%d"
	}
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query(`SELECT strftime(?, timestamp) as period, COUNT(*), `+tokensSavedSum+`, COALESCE(AVG(duration_ms),0)
			FROM queries WHERE project_path = ? AND timestamp >= datetime('now', '-' || ? || ' days') GROUP BY period ORDER BY period ASC`, format, pid, days)
	} else {
		rows, err = db.DB.Query(`SELECT strftime(?, timestamp) as period, COUNT(*), `+tokensSavedSum+`, COALESCE(AVG(duration_ms),0)
			FROM queries WHERE timestamp >= datetime('now', '-' || ? || ' days') GROUP BY period ORDER BY period ASC`, format, days)
	}
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode([]interface{}{})
		return
	}
	defer rows.Close()
	type point struct {
		Timestamp     string  `json:"timestamp"`
		Queries       int     `json:"queries"`
		TokensSaved   int     `json:"tokens_saved"`
		AvgDurationMs float64 `json:"avg_duration_ms"`
	}
	var points []point
	for rows.Next() {
		var p point
		rows.Scan(&p.Timestamp, &p.Queries, &p.TokensSaved, &p.AvgDurationMs)
		points = append(points, p)
	}
	if points == nil {
		points = []point{}
	}
	json.NewEncoder(w).Encode(points)
}

func handleSymbolChartPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query("SELECT kind, COUNT(*) as count FROM symbols WHERE project_path = ? GROUP BY kind ORDER BY count DESC", pid)
	} else {
		rows, err = db.DB.Query("SELECT kind, COUNT(*) as count FROM symbols GROUP BY kind ORDER BY count DESC")
	}
	if err != nil {
		components.BarChart(nil, 0).Render(r.Context(), w)
		return
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var kind string
		var count int
		rows.Scan(&kind, &count)
		items = append(items, components.BarItem{Label: kind, Value: count})
	}
	components.BarChart(items, 0).Render(r.Context(), w)
}

func handleLanguageChartPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	q := `SELECT CASE
		WHEN file LIKE '%.py' THEN 'Python' WHEN file LIKE '%.go' THEN 'Go'
		WHEN file LIKE '%.js' THEN 'JavaScript' WHEN file LIKE '%.jsx' THEN 'JSX'
		WHEN file LIKE '%.ts' THEN 'TypeScript' WHEN file LIKE '%.tsx' THEN 'TSX'
		WHEN file LIKE '%.sh' THEN 'Bash' WHEN file LIKE '%.fish' THEN 'Fish'
		ELSE 'Other' END as language, COUNT(*) as symbols FROM symbols`
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query(q+" WHERE project_path = ? GROUP BY language ORDER BY symbols DESC", pid)
	} else {
		rows, err = db.DB.Query(q + " GROUP BY language ORDER BY symbols DESC")
	}
	if err != nil {
		components.BarChart(nil, 3).Render(r.Context(), w)
		return
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var lang string
		var symbols int
		rows.Scan(&lang, &symbols)
		items = append(items, components.BarItem{Label: lang, Value: symbols})
	}
	components.BarChart(items, 3).Render(r.Context(), w)
}

func handleToolChartPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	components.ToolPerformancePanel(queryToolStats(pid)).Render(r.Context(), w)
}

func handleImportChartPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query("SELECT target, COUNT(*) as count FROM edges WHERE project_path = ? GROUP BY target ORDER BY count DESC LIMIT 20", pid)
	} else {
		rows, err = db.DB.Query("SELECT target, COUNT(*) as count FROM edges GROUP BY target ORDER BY count DESC LIMIT 20")
	}
	if err != nil {
		components.BarChart(nil, 5).Render(r.Context(), w)
		return
	}
	defer rows.Close()
	var items []components.BarItem
	for rows.Next() {
		var target string
		var count int
		rows.Scan(&target, &count)
		items = append(items, components.BarItem{Label: target, Value: count})
	}
	components.BarChart(items, 5).Render(r.Context(), w)
}

func handleRecentPartial(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	lim := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			lim = parsed
		}
	}
	mcp, indexing := buildRecentQueries(pid, lim)
	components.RecentPanel(mcp, indexing).Render(r.Context(), w)
}

func handleSettingsPartial(w http.ResponseWriter, r *http.Request) {
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
				Refreshing:  docs.IsRefreshing(s.ID),
			})
		}
	}

	data := components.SettingsData{
		IdleUnloadMinutes:       idleMinutes,
		WatcherIgnoreGlobs:      watcherIgn,
		IndexLogFiles:           indexLog,
		LogRetentionEnabled:     logRetentionEn,
		LogRetentionRoots:       logRoots,
		LogRetentionMaxAgeDays:  logRetentionMaxAge,
		LogRetentionMaxTotalMB:  logRetentionMaxMB,
		LogRetentionDryRun:      logDry,
		LogRetentionLastRun:     logLast,
		Projects:                projects,
		Agents:                  agents,
		DocSources:              docSources,
	}
	PopulateEmbedSettings(settings, &data)
	applyActiveEmbedderSettings(&data)
	loadEmbedModels(&data)
	components.Settings(data).Render(r.Context(), w)
}

func loadProjects(pid string) []components.Project {
	type symCount struct {
		symbols int
		files   int
	}
	symCounts := map[string]symCount{}
	symRows, err := db.DB.Query("SELECT project_path, COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path IS NOT NULL GROUP BY project_path")
	if err == nil {
		defer symRows.Close()
		for symRows.Next() {
			var pp string
			var sc symCount
			symRows.Scan(&pp, &sc.symbols, &sc.files)
			symCounts[pp] = sc
		}
	}
	rows, err := db.DB.Query("SELECT DISTINCT project_path, COUNT(*) FROM queries WHERE project_path IS NOT NULL AND project_path != '' AND project_path != '.' GROUP BY project_path")
	if err != nil {
		return nil
	}
	defer rows.Close()
	seen := map[string]bool{}
	var ps []components.Project
	for rows.Next() {
		var p string
		var c int
		rows.Scan(&p, &c)
		seen[p] = true
		sc := symCounts[p]
		ps = append(ps, components.Project{
			Path:        p,
			Name:        filepath.Base(p),
			QueryCount:  c,
			SymbolCount: sc.symbols,
			FileCount:   sc.files,
			Pinned:      db.IsPinnedProject(p),
		})
	}
	for pp, sc := range symCounts {
		if seen[pp] {
			continue
		}
		ps = append(ps, components.Project{
			Path:        pp,
			Name:        filepath.Base(pp),
			SymbolCount: sc.symbols,
			FileCount:   sc.files,
			Pinned:      db.IsPinnedProject(pp),
		})
	}
	sort.Slice(ps, func(i, j int) bool {
		ni, nj := strings.ToLower(ps[i].Name), strings.ToLower(ps[j].Name)
		if ni != nj {
			return ni < nj
		}
		return ps[i].Path < ps[j].Path
	})
	return ps
}
