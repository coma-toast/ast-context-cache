package dashboard

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
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
)

var serverStartTime = time.Now()

var (
	projectsCacheMu sync.Mutex
	projectsCache   []components.Project
	projectsCacheAt time.Time
)

const projectsCacheTTL = 15 * time.Second

func loadProjects(pid string) []components.Project {
	_ = pid
	projectsCacheMu.Lock()
	if len(projectsCache) > 0 && time.Since(projectsCacheAt) < projectsCacheTTL {
		out := make([]components.Project, len(projectsCache))
		copy(out, projectsCache)
		projectsCacheMu.Unlock()
		return out
	}
	projectsCacheMu.Unlock()
	ps := loadProjectsFresh()
	projectsCacheMu.Lock()
	projectsCache = ps
	projectsCacheAt = time.Now()
	projectsCacheMu.Unlock()
	return ps
}

func loadProjectsForPage() []components.Project {
	projectsCacheMu.Lock()
	if len(projectsCache) > 0 {
		out := make([]components.Project, len(projectsCache))
		copy(out, projectsCache)
		stale := time.Since(projectsCacheAt) >= projectsCacheTTL
		projectsCacheMu.Unlock()
		if stale {
			go func() { loadProjects("") }()
		}
		return out
	}
	projectsCacheMu.Unlock()
	go func() { loadProjects("") }()
	return nil
}

func loadProjectsFresh() []components.Project {
	type symCount struct {
		symbols int
		files   int
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	symCounts := map[string]symCount{}
	symRows, err := db.DB.QueryContext(ctx, "SELECT project_path, COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path IS NOT NULL GROUP BY project_path")
	if err == nil {
		defer symRows.Close()
		for symRows.Next() {
			var pp string
			var sc symCount
			symRows.Scan(&pp, &sc.symbols, &sc.files)
			symCounts[pp] = sc
		}
	}
	queryCounts := map[string]int{}
	rows, err := db.DB.QueryContext(ctx, "SELECT DISTINCT project_path, COUNT(*) FROM queries WHERE project_path IS NOT NULL AND project_path != '' AND project_path != '.' GROUP BY project_path")
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var p string
			var c int
			rows.Scan(&p, &c)
			queryCounts[p] = c
		}
	}
	allPaths := map[string]bool{}
	for pp := range symCounts {
		allPaths[pp] = true
	}
	for pp := range queryCounts {
		allPaths[pp] = true
	}
	for _, pp := range projectmeta.DiscoverPaths() {
		allPaths[pp] = true
	}
	pinned := map[string]bool{}
	for _, p := range db.GetPinnedProjects() {
		pinned[p] = true
	}
	var ps []components.Project
	for pp := range allPaths {
		if projectmeta.IsExcluded(pp) {
			continue
		}
		meta := projectmeta.Enrich(pp)
		sc := symCounts[pp]
		label := meta.Label
		if label == "" {
			label = filepath.Base(pp)
		}
		ps = append(ps, components.Project{
			Path:        pp,
			Name:        meta.RepoName,
			Label:       label,
			Workspace:   meta.Workspace,
			Branch:      meta.Branch,
			RepoKey:     meta.RepoKey,
			QueryCount:  queryCounts[pp],
			SymbolCount: sc.symbols,
			FileCount:   sc.files,
			Pinned:      pinned[pp],
		})
	}
	sort.Slice(ps, func(i, j int) bool {
		li, lj := strings.ToLower(ps[i].Label), strings.ToLower(ps[j].Label)
		if li != lj {
			return li < lj
		}
		return ps[i].Path < ps[j].Path
	})
	return ps
}

func handleDashboardPage(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	projects := loadProjectsForPage()
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
	fillVirtualContextStats(&s, pid)
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
		QueueWorkers:   eq.Workers,
		QueueThroughput: eq.Throughput,
		QueueQueued:     eq.Queued,
		QueuePending:     eq.Pending,
		QueuePendingPeak: eq.PendingPeak,
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

func handleMemoryPartial(w http.ResponseWriter, r *http.Request) {
	components.MemoryPanel(buildMemory(r.URL.Query().Get("project_id"), parseDocSourcesPageQuery(r))).Render(r.Context(), w)
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
	logs, logPath, logTrunc, logOpts := buildRecentLogsForDashboard()
	components.RecentPanel(mcp, indexing, logs, logPath, logTrunc, logOpts).Render(r.Context(), w)
}

func handleRecentLogsPartial(w http.ResponseWriter, r *http.Request) {
	logs, logPath, logTrunc, logOpts := buildRecentLogsForDashboard()
	components.RecentLogs(logs, logPath, logTrunc, logOpts).Render(r.Context(), w)
}

func handleSettingsPartial(w http.ResponseWriter, r *http.Request) {
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
		WatcherIgnoreGlobs:      watcherIgn,
		ProjectExcludePaths:     projectExclude,
		IndexLogFiles:           indexLog,
		LogRetentionEnabled:     logRetentionEn,
		LogRetentionRoots:       logRoots,
		LogRetentionMaxAgeDays:  logRetentionMaxAge,
		LogRetentionMaxTotalMB:  logRetentionMaxMB,
		LogRetentionDryRun:      logDry,
		LogRetentionLastRun:     logLast,
		QueryRetentionEnabled:   queryRetentionEn,
		QueryRetentionMaxAgeDays: queryRetentionMaxAge,
		QueryRetentionLastRun:   queryRetentionLast,
		Projects:                projects,
		Agents:                  agents,
		EmbedWorkerMax:          embedqueue.MaxWorkers(),
	}
	PopulateEmbedSettings(settings, &data)
	populateContextSettings(settings, &data)
	applyActiveEmbedderSettings(&data)
	loadEmbedModels(&data)
	components.Settings(data).Render(r.Context(), w)
}
