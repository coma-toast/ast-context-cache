package dashboard

import (
	"crypto/sha256"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

//go:embed static/*
var staticAssets embed.FS
var staticFS, _ = fs.Sub(staticAssets, "static")

func NewHandler(_ string) http.Handler {
	mux := http.NewServeMux()

	// JSON APIs (kept for backward compatibility + MCP tools)
	mux.HandleFunc("/api/stats", handleStats)
	mux.HandleFunc("/api/tools", handleTools)
	mux.HandleFunc("/api/recent", handleRecent)
	mux.HandleFunc("/api/projects", handleProjects)
	mux.HandleFunc("/api/reset", handleReset)
	mux.HandleFunc("/api/delete", handleDeleteProject)
	mux.HandleFunc("/api/reset-project", handleResetProject)
	mux.HandleFunc("/api/stop-watcher", handleStopWatcher)
	mux.HandleFunc("/api/start-watcher", handleStartWatcher)
	mux.HandleFunc("/api/index-project", handleIndexProject)
	mux.HandleFunc("/api/delete-watcher", handleDeleteWatcher)
	mux.HandleFunc("/api/timeseries", handleTimeseries)
	mux.HandleFunc("/api/index-stats", handleIndexStats)
	mux.HandleFunc("/api/symbol-kinds", handleSymbolKinds)
	mux.HandleFunc("/api/language-stats", handleLanguageStats)
	mux.HandleFunc("/api/top-imports", handleTopImports)
	mux.HandleFunc("/api/watcher-status", handleWatcherStatus)
	mux.HandleFunc("/api/vector-stats", handleVectorStats)
	mux.HandleFunc("/api/settings", handleSettings)
	mux.HandleFunc("/api/settings/embed", handleEmbedSettings)
	mux.HandleFunc("/api/embedder/test", handleEmbedderTest)
	mux.HandleFunc("/api/embedder/retry", handleEmbedderRetry)
	mux.HandleFunc("/api/embedder/dismiss-alert", handleEmbedderDismissAlert)
	mux.HandleFunc("/api/embedder/models", handleEmbedModels)
	mux.HandleFunc("/api/embedder/docker-models", handleDockerModels)
	mux.HandleFunc("/api/pin-project", handlePinProject)
	mux.HandleFunc("/api/embed-workers", handleEmbedWorkers)
	mux.HandleFunc("/api/embed-aux-workers", handleEmbedAuxWorkers)
	mux.HandleFunc("/api/agent-configs", handleAgentConfigs)
	mux.HandleFunc("/api/agent-install", handleAgentInstall)
	mux.HandleFunc("/api/agent-uninstall", handleAgentUninstall)
	mux.HandleFunc("/api/system-resources", handleSystemResources)
	mux.HandleFunc("/api/doc-sources", handleDocSources)
	mux.HandleFunc("/api/context-stats", handleContextStats)
	mux.HandleFunc("/api/kv-repair-stats", handleKvRepairStats)
	mux.HandleFunc("/api/flush-context", handleFlushContextAPI)

	// WebSocket
	mux.HandleFunc("/ws", handleWS)

	// HTML partials (htmx targets)
	mux.HandleFunc("/partials/stats", handleStatsPartial)
	mux.HandleFunc("/partials/index-health", handleIndexHealthPartial)
	mux.HandleFunc("/partials/memory", handleMemoryPartial)
	mux.HandleFunc("/partials/recent", handleRecentPartial)
	mux.HandleFunc("/partials/recent-logs", handleRecentLogsPartial)
	mux.HandleFunc("/partials/charts/symbols", handleSymbolChartPartial)
	mux.HandleFunc("/partials/charts/languages", handleLanguageChartPartial)
	mux.HandleFunc("/partials/charts/tools", handleToolChartPartial)
	mux.HandleFunc("/partials/charts/imports", handleImportChartPartial)
	mux.HandleFunc("/partials/settings", handleSettingsPartial)
	mux.HandleFunc("/partials/activity-data", handleActivityDataPartial)
	mux.HandleFunc("/partials/health", handleHealthPartial)

	// Aliases for React dashboard
	mux.HandleFunc("/dashboard/partials/health", handleHealthPartial)
	mux.HandleFunc("/dashboard/partials/activity", handleActivityPartial)

	// Static assets (CSS, JS)
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))

	// Dashboard page (templ-rendered)
	mux.HandleFunc("/", handleDashboardPage)

	return mux
}

type stats struct {
	TotalQueries       int     `json:"total_queries"`
	TotalSessions      int     `json:"total_sessions"`
	TotalChars         int     `json:"total_chars"`
	TotalTokensSaved   int     `json:"total_tokens_saved"`
	AvgDurationMs      float64 `json:"avg_duration_ms"`
	TodayQueries       int     `json:"today_queries"`
	TodayTokensSaved   int     `json:"today_tokens_saved"`
	TodaySessions      int     `json:"today_sessions"`
	TodayAvgDurationMs float64 `json:"today_avg_duration_ms"`
	WindowDays         int     `json:"window_days"`
}

func handleStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	var s stats
	s.WindowDays = StatsWindowDays
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	tokensSavedSum := "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN tokens_saved ELSE 0 END),0)"
	statsSel := "SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), " + tokensSavedSum + " FROM queries WHERE "
	where, args := statsQueriesWhere(pid)
	db.DB.QueryRow(statsSel+where, args...).
		Scan(&s.TotalQueries, &s.TotalSessions, &s.TotalChars, &s.AvgDurationMs, &s.TotalTokensSaved)
	var today components.Stats
	fillTodayStats(pid, todayStart, tomorrowStart, &today)
	s.TodayQueries = today.TodayQueries
	s.TodayTokensSaved = today.TodayTokens
	s.TodaySessions = today.TodaySessions
	s.TodayAvgDurationMs = today.TodayAvgDurationMs
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s)
}

func handleTools(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	stats := queryToolStats(pid)
	w.Header().Set("Content-Type", "application/json")
	if stats == nil {
		json.NewEncoder(w).Encode([]any{})
		return
	}
	json.NewEncoder(w).Encode(stats)
}

func handleRecent(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	lim := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := fmt.Sscanf(l, "%d", &lim); err != nil || parsed != 1 || lim <= 0 {
			lim = 50
		}
	}
	if lim > 500 {
		lim = 500
	}
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query("SELECT timestamp, tool_name, result_chars, duration_ms, project_path, COALESCE(error,''), COALESCE(arguments,''), COALESCE(tokens_saved,0), COALESCE(file_baseline_tokens,0) FROM queries WHERE project_path = ? ORDER BY timestamp DESC LIMIT ?", pid, lim)
	} else {
		rows, err = db.DB.Query("SELECT timestamp, tool_name, result_chars, duration_ms, project_path, COALESCE(error,''), COALESCE(arguments,''), COALESCE(tokens_saved,0), COALESCE(file_baseline_tokens,0) FROM queries ORDER BY timestamp DESC LIMIT ?", lim)
	}
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	qs := []map[string]interface{}{}
	for rows.Next() {
		var t, n, pp, errMsg, argsJSON string
		var rc, saved, fileBaseline int
		var dm float64
		rows.Scan(&t, &n, &rc, &dm, &pp, &errMsg, &argsJSON, &saved, &fileBaseline)
		entry := map[string]interface{}{"timestamp": t, "tool_name": n, "result_chars": rc, "duration_ms": dm, "project_path": pp, "error": errMsg, "tokens_saved": saved, "file_baseline_tokens": fileBaseline}
		var parsed map[string]interface{}
		if json.Unmarshal([]byte(argsJSON), &parsed) == nil {
			if a, ok := parsed["arguments"].(map[string]interface{}); ok {
				parsed = a
			}
			if q, ok := parsed["query"].(string); ok {
				entry["query"] = q
			}
			if m, ok := parsed["mode"].(string); ok && m != "" {
				entry["mode"] = m
			}
			if tb, ok := parsed["token_budget"].(float64); ok && tb > 0 {
				entry["token_budget"] = int(tb)
			}
		}
		qs = append(qs, entry)
	}
	json.NewEncoder(w).Encode(qs)
}

func handleProjects(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
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
	rows, err := db.DB.Query("SELECT DISTINCT project_path, COUNT(*) FROM queries WHERE project_path IS NOT NULL GROUP BY project_path")
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	var ps []map[string]interface{}
	for rows.Next() {
		var p string
		var c int
		rows.Scan(&p, &c)
		sc := symCounts[p]
		ps = append(ps, map[string]interface{}{
			"path": p, "name": filepath.Base(p), "query_count": c,
			"symbol_count": sc.symbols, "file_count": sc.files,
		})
	}
	if ps == nil {
		ps = []map[string]interface{}{}
	}
	json.NewEncoder(w).Encode(ps)
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	var req map[string]string
	json.NewDecoder(r.Body).Decode(&req)
	projectPath := req["project_path"]

	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")

	if projectPath == "all" {
		_, err := db.DB.Exec("DELETE FROM symbols")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		db.DB.Exec("DELETE FROM edges")
		db.DB.Exec("DELETE FROM indexed_files")
		db.DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
		db.EnsureFTSTriggers()
		cache.GlobalCache.ClearAll()
		go db.Compact()
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "message": "All indexed data cleared"})
	} else if projectPath != "" {
		projectPath = watcher.NormalizeProjectPath(projectPath)
		embedqueue.RemoveProject(projectPath)
		_, err := db.DB.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		db.DB.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
		db.DB.Exec("DELETE FROM indexed_files WHERE project_path = ?", projectPath)
		db.DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
		db.EnsureFTSTriggers()
		cache.GlobalCache.ClearProject(projectPath)
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "project_path": projectPath})
	} else {
		json.NewEncoder(w).Encode(map[string]string{"error": "project_path required"})
	}
}

func handleDeleteProject(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	var req map[string]string
	json.NewDecoder(r.Body).Decode(&req)
	projectPath := watcher.NormalizeProjectPath(req["project_path"])
	if projectPath == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "project_path required"})
		return
	}
	embedqueue.RemoveProject(projectPath)
	db.DB.Exec("DELETE FROM queries WHERE project_path = ?", projectPath)
	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")
	db.DB.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM indexed_files WHERE project_path = ?", projectPath)
	db.DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	db.EnsureFTSTriggers()
	cache.GlobalCache.ClearProject(projectPath)
	go db.Compact()
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "project_path": projectPath})
}

func handleTimeseries(w http.ResponseWriter, r *http.Request) {
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
	tokensSavedSum := "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN tokens_saved ELSE 0 END),0)"
	if pid != "" {
		rows, err = db.DB.Query(`SELECT strftime(?, timestamp) as period, COUNT(*), `+tokensSavedSum+`, COALESCE(AVG(duration_ms),0)
			FROM queries WHERE project_path = ? AND timestamp >= datetime('now', '-' || ? || ' days') GROUP BY period ORDER BY period ASC`, format, pid, days)
	} else {
		rows, err = db.DB.Query(`SELECT strftime(?, timestamp) as period, COUNT(*), `+tokensSavedSum+`, COALESCE(AVG(duration_ms),0)
			FROM queries WHERE timestamp >= datetime('now', '-' || ? || ' days') GROUP BY period ORDER BY period ASC`, format, days)
	}
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
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

func handleIndexStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	var totalSymbols, totalFiles, totalEdges int
	if pid != "" {
		db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", pid).Scan(&totalSymbols, &totalFiles)
		db.DB.QueryRow("SELECT COUNT(*) FROM edges WHERE project_path = ?", pid).Scan(&totalEdges)
	} else {
		db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&totalSymbols, &totalFiles)
		db.DB.QueryRow("SELECT COUNT(*) FROM edges").Scan(&totalEdges)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_symbols":    totalSymbols,
		"total_files":      totalFiles,
		"total_edges":      totalEdges,
		"total_vectors":    search.Cache.Count(pid),
		"vector_memory_mb": search.Cache.MemoryMB(),
	})
}

func handleSymbolKinds(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query("SELECT kind, COUNT(*) as count FROM symbols WHERE project_path = ? GROUP BY kind ORDER BY count DESC", pid)
	} else {
		rows, err = db.DB.Query("SELECT kind, COUNT(*) as count FROM symbols GROUP BY kind ORDER BY count DESC")
	}
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	kinds := []map[string]interface{}{}
	for rows.Next() {
		var kind string
		var count int
		rows.Scan(&kind, &count)
		kinds = append(kinds, map[string]interface{}{"kind": kind, "count": count})
	}
	json.NewEncoder(w).Encode(kinds)
}

func handleLanguageStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	q := `SELECT CASE
		WHEN file LIKE '%.py' THEN 'Python' WHEN file LIKE '%.go' THEN 'Go'
		WHEN file LIKE '%.js' THEN 'JavaScript' WHEN file LIKE '%.jsx' THEN 'JSX'
		WHEN file LIKE '%.ts' THEN 'TypeScript' WHEN file LIKE '%.tsx' THEN 'TSX'
		WHEN file LIKE '%.sh' THEN 'Bash' WHEN file LIKE '%.fish' THEN 'Fish'
		ELSE 'Other' END as language, COUNT(DISTINCT file) as files, COUNT(*) as symbols FROM symbols`
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query(q+" WHERE project_path = ? GROUP BY language ORDER BY symbols DESC", pid)
	} else {
		rows, err = db.DB.Query(q + " GROUP BY language ORDER BY symbols DESC")
	}
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	langs := []map[string]interface{}{}
	for rows.Next() {
		var lang string
		var files, symbols int
		rows.Scan(&lang, &files, &symbols)
		langs = append(langs, map[string]interface{}{"language": lang, "files": files, "symbols": symbols})
	}
	json.NewEncoder(w).Encode(langs)
}

func handleTopImports(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	var rows *sql.Rows
	var err error
	if pid != "" {
		rows, err = db.DB.Query("SELECT target, COUNT(*) as count FROM edges WHERE project_path = ? GROUP BY target ORDER BY count DESC LIMIT 20", pid)
	} else {
		rows, err = db.DB.Query("SELECT target, COUNT(*) as count FROM edges GROUP BY target ORDER BY count DESC LIMIT 20")
	}
	if err != nil {
		json.NewEncoder(w).Encode([]map[string]interface{}{})
		return
	}
	defer rows.Close()
	imports := []map[string]interface{}{}
	for rows.Next() {
		var target string
		var count int
		rows.Scan(&target, &count)
		imports = append(imports, map[string]interface{}{"target": target, "count": count})
	}
	json.NewEncoder(w).Encode(imports)
}

func handleWatcherStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(watcher.GetStatus())
}

func handleVectorStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")

	totalVectors := search.Cache.Count(pid)
	memoryMB := search.Cache.MemoryMB()

	var dbVectors int
	if pid != "" {
		db.DB.QueryRow("SELECT COUNT(*) FROM vectors WHERE project_path = ?", pid).Scan(&dbVectors)
	} else {
		db.DB.QueryRow("SELECT COUNT(*) FROM vectors").Scan(&dbVectors)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_vectors": totalVectors,
		"db_vectors":    dbVectors,
		"memory_mb":     memoryMB,
		"dimensions":    search.VectorDims,
		"cache_loaded":  search.Cache.IsLoaded(),
	})
}

func handlePinProject(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "POST required"})
		return
	}
	var req struct {
		ProjectPath string `json:"project_path"`
		Pinned      bool   `json:"pinned"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	if req.ProjectPath == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "project_path required"})
		return
	}
	if err := db.TogglePinnedProject(req.ProjectPath, req.Pinned); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	realtime.Notify(realtime.SettingsChanged | realtime.IndexHealth)
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "pinned": req.Pinned})
}

func parseEmbedWorkersRequest(r *http.Request) (delta int, count *int, err error) {
	ct := r.Header.Get("Content-Type")
	if strings.Contains(ct, "application/json") {
		var req struct {
			Delta int  `json:"delta"`
			Count *int `json:"count"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return 0, nil, err
		}
		return req.Delta, req.Count, nil
	}
	if err := r.ParseForm(); err != nil {
		return 0, nil, err
	}
	if c := r.FormValue("count"); c != "" {
		n, err := strconv.Atoi(c)
		if err != nil {
			return 0, nil, fmt.Errorf("count: %w", err)
		}
		return 0, &n, nil
	}
	delta, _ = strconv.Atoi(r.FormValue("delta"))
	return delta, nil, nil
}

func embedWorkersSnapshot() map[string]interface{} {
	return map[string]interface{}{
		"workers":     embedqueue.WorkerCount(),
		"max_workers": embedqueue.MaxWorkers(),
		"live":        embedqueue.WorkerLive(),
	}
}

func handleEmbedWorkers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodGet {
		out := embedWorkersSnapshot()
		out["status"] = "ok"
		json.NewEncoder(w).Encode(out)
		return
	}
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "GET or POST required"})
		return
	}
	delta, count, err := parseEmbedWorkersRequest(r)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	var n int
	if count != nil {
		n, err = embedqueue.SetWorkerCount(*count)
	} else if delta != 0 {
		n, err = embedqueue.AdjustWorkers(delta)
	} else {
		n = embedqueue.WorkerCount()
	}
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	realtime.Notify(realtime.IndexHealth | realtime.HealthBar)
	out := embedWorkersSnapshot()
	out["status"] = "ok"
	out["workers"] = n
	json.NewEncoder(w).Encode(out)
}

func embedAuxWorkersSnapshot() map[string]interface{} {
	backend, model := embedder.AuxSnapshot()
	return map[string]interface{}{
		"workers":     embedqueue.AuxWorkerCount(),
		"max_workers": embedqueue.AuxMaxWorkers(),
		"live":        embedqueue.AuxWorkerLive(),
		"backend":     backend,
		"model":       model,
		"enabled":     backend != "" && !embedder.AuxSharesPrimary(),
	}
}

func handleEmbedAuxWorkers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if embedder.AuxSharesPrimary() {
		if r.Method == http.MethodGet {
			out := embedAuxWorkersSnapshot()
			out["status"] = "ok"
			json.NewEncoder(w).Encode(out)
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"error": "aux pool requires a different backend than primary (set EMBED_AUX_BACKEND, default onnx)"})
		return
	}
	if r.Method == http.MethodGet {
		out := embedAuxWorkersSnapshot()
		out["status"] = "ok"
		json.NewEncoder(w).Encode(out)
		return
	}
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "GET or POST required"})
		return
	}
	delta, count, err := parseEmbedWorkersRequest(r)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	var n int
	if count != nil {
		n, err = embedqueue.SetAuxWorkerCount(*count)
	} else if delta != 0 {
		n, err = embedqueue.AdjustAuxWorkers(delta)
	} else {
		n = embedqueue.AuxWorkerCount()
	}
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	realtime.Notify(realtime.IndexHealth | realtime.HealthBar)
	out := embedAuxWorkersSnapshot()
	out["status"] = "ok"
	out["workers"] = n
	json.NewEncoder(w).Encode(out)
}

func handleSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method == "POST" {
		ct := r.Header.Get("Content-Type")
		var key, value string
		if strings.Contains(ct, "application/json") {
			var req map[string]string
			json.NewDecoder(r.Body).Decode(&req)
			key, value = req["key"], req["value"]
		} else {
			r.ParseForm()
			key, value = r.FormValue("key"), r.FormValue("value")
		}
		if key == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "key required"})
			return
		}
		if key == "embed_worker_max" {
			n, err := strconv.Atoi(value)
			if err != nil || n < 1 || n > embedqueue.AbsoluteMaxWorkers {
				json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("embed_worker_max must be 1–%d", embedqueue.AbsoluteMaxWorkers)})
				return
			}
		}
		if key == "embed_aux_worker_max" {
			n, err := strconv.Atoi(value)
			if err != nil || n < 1 || n > 32 {
				json.NewEncoder(w).Encode(map[string]string{"error": "embed_aux_worker_max must be 1–32"})
				return
			}
		}
		if key == "EMBED_AUX_WORKERS" {
			max := embedqueue.AuxMaxWorkers()
			n, err := strconv.Atoi(value)
			if err != nil || n < embedqueue.MinWorkers || n > max {
				json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("EMBED_AUX_WORKERS must be %d–%d", embedqueue.MinWorkers, max)})
				return
			}
		}
		if key == "EMBED_AUX_BACKEND" {
			value = normalizeEmbedBackendValue(value)
		}
		if key == "EMBED_BACKEND" {
			value = normalizeEmbedBackendValue(value)
			oldBackend := components.EmbedBackendUI(db.GetSetting("EMBED_BACKEND", ""))
			newBackend := components.EmbedBackendUI(value)
			if err := switchEmbedBackend(oldBackend, newBackend, value); err != nil {
				json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
				return
			}
			writeSettingsOK(w, map[string]string{"key": key, "value": value}, true, realtime.SettingsChanged)
			return
		}
		if err := db.SetSetting(key, value); err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		if key == "watcher_ignore_globs" {
			ignorepatterns.InvalidateCache()
		}
		if key == "project_exclude_paths" {
			projectmeta.InvalidateExcludeCache()
		}
		if key == "embed_worker_max" {
			if err := embedqueue.ClampWorkersToMax(); err != nil {
				json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
				return
			}
			invalidatePartialCache("#index-health")
		}
		if key == "embed_aux_worker_max" {
			if err := embedqueue.ClampAuxWorkersToMax(); err != nil {
				json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
				return
			}
			invalidatePartialCache("#index-health")
		}
		if key == "EMBED_AUX_WORKERS" {
			n, _ := strconv.Atoi(value)
			if _, err := embedqueue.SetAuxWorkerCount(n); err != nil {
				log.Printf("dashboard: set aux workers: %v", err)
			}
			invalidatePartialCache("#index-health")
		}
		if key == "EMBED_AUX_BACKEND" {
			if embedder.AuxRuntimeReady() {
				if err := embedder.ReloadAuxRuntime(); err != nil {
					json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
					return
				}
				embedqueue.SetAuxEmbedder(embedder.RawAux())
			}
			invalidatePartialCache("#index-health")
		}
		onEmbedSettingChanged(key)
		mask := realtime.SettingsChanged
		reloadEmbed := embedder.IsSettingKey(key)
		if reloadEmbed {
			mask = realtime.IndexHealth | realtime.HealthBar
		} else if key == "idle_unload_minutes" {
			mask |= realtime.IndexHealth | realtime.HealthBar
		} else if key == "project_exclude_paths" {
			mask |= realtime.IndexHealth
		} else if key == "embed_worker_max" {
			mask |= realtime.IndexHealth | realtime.HealthBar
		} else if key == "embed_aux_worker_max" || key == "EMBED_AUX_WORKERS" || key == "EMBED_AUX_BACKEND" {
			mask |= realtime.IndexHealth | realtime.HealthBar
		}
		writeSettingsOK(w, map[string]string{"key": key, "value": value}, reloadEmbed, mask)
		return
	}
	defaults := map[string]string{
		"idle_unload_minutes":         "1",
		"watcher_ignore_globs":         ignorepatterns.JSON(ignorepatterns.DefaultGlobs),
		"project_exclude_paths":        "[]",
		"index_log_files":              "false",
		"log_retention_enabled":        "false",
		"log_retention_roots":          "[]",
		"log_retention_max_age_days":   "0",
		"log_retention_max_total_mib":  "0",
		"log_retention_dry_run":        "false",
		"query_retention_enabled":      "true",
		"query_retention_max_age_days":   "90",
		"EMBED_BACKEND":                "",
		"MODEL_DIR":                    "",
		"EMBED_HTTP_URL":               "",
		"EMBED_HTTP_BEARER":            "",
		"OLLAMA_HOST":                  "",
		"OLLAMA_EMBED_MODEL":           "",
		"EMBED_OPENAI_BASE_URL":        "",
		"EMBED_OPENAI_API_KEY":         "",
		"EMBED_OPENAI_MODEL":           "",
		"EMBED_OPENAI_DIMENSIONS":      "",
		"EMBED_DOCKER_URL":             "",
		"EMBED_DOCKER_MODEL":           "",
		"EMBED_DOCKER_DIMENSIONS":      "",
		"context_max_notes_session":     "50",
		"context_max_tokens_session":    "32000",
		"context_max_notes_global":      "500",
		"context_max_tokens_global":     "200000",
		"context_limit_policy":          "reject",
		"embed_worker_max":              strconv.Itoa(embedqueue.DefaultMaxWorkers),
		"embed_aux_worker_max":          strconv.Itoa(embedqueue.DefaultAuxMaxWorkers),
		"EMBED_AUX_WORKERS":             "0",
		"EMBED_AUX_BACKEND":             "onnx",
		"dashboard_log_tail_lines":      "200",
		"dashboard_log_line_chars":      "500",
	}
	settings := db.GetAllSettings()
	for k, v := range defaults {
		if _, ok := settings[k]; !ok {
			settings[k] = v
		}
	}
	json.NewEncoder(w).Encode(settings)
}

type AgentInfo struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	GlobalPath  string `json:"global_path"`
	ProjectPath string `json:"project_path"`
	Description string `json:"description"`
}

var supportedAgents = []AgentInfo{
	{"cursor", "Cursor", "~/.cursor/mcp.json", ".cursor/mcp.json", "MCP server config for Cursor IDE"},
	{"opencode", "OpenCode", "~/.config/opencode/opencode.jsonc", "opencode.jsonc", "MCP settings for OpenCode"},
	{"claude_code", "Claude Code", "~/.claude.json", "CLAUDE.md", "Claude Code instructions file"},
	{"claude_desktop", "Claude Desktop", "~/Library/Application Support/Claude/claude_desktop_config.json", ".claude_desktop_config.json", "Claude Desktop MCP config"},
}

func handleAgentConfigs(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	configs, err := db.GetAgentConfigs()
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	type response struct {
		Agents    []AgentInfo      `json:"agents"`
		Installed []db.AgentConfig `json:"installed"`
	}
	json.NewEncoder(w).Encode(response{Agents: supportedAgents, Installed: configs})
}

func handleAgentInstall(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}

	var req struct {
		AgentType string `json:"agent_type"`
		IsGlobal  bool   `json:"is_global"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	if req.AgentType == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "agent_type required"})
		return
	}

	var agent AgentInfo
	for _, a := range supportedAgents {
		if a.Type == req.AgentType {
			agent = a
			break
		}
	}
	if agent.Type == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "unknown agent type"})
		return
	}

	home := os.Getenv("HOME")
	var installPath string
	if req.IsGlobal {
		installPath = agent.GlobalPath
	} else {
		installPath = agent.ProjectPath
	}

	fullPath := installPath
	if strings.HasPrefix(fullPath, "~/") {
		fullPath = filepath.Join(home, strings.TrimPrefix(fullPath, "~"))
	}

	instructions := generateAgentInstructions(req.AgentType)
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(instructions)))

	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "failed to create dir: " + err.Error()})
		return
	}

	if err := os.WriteFile(fullPath, []byte(instructions), 0644); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "failed to write file: " + err.Error()})
		return
	}

	if err := db.AddAgentConfig(req.AgentType, installPath, req.IsGlobal, hash); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "failed to save config: " + err.Error()})
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "installed",
		"agent_type":   req.AgentType,
		"path":         fullPath,
		"is_global":    req.IsGlobal,
		"instructions": instructions,
	})
}

func handleAgentUninstall(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}

	var req struct {
		AgentType string `json:"agent_type"`
		IsGlobal  bool   `json:"is_global"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	home := os.Getenv("HOME")
	var installPath string
	if req.IsGlobal {
		for _, a := range supportedAgents {
			if a.Type == req.AgentType {
				installPath = a.GlobalPath
				break
			}
		}
	} else {
		for _, a := range supportedAgents {
			if a.Type == req.AgentType {
				installPath = a.ProjectPath
				break
			}
		}
	}

	fullPath := installPath
	if strings.HasPrefix(fullPath, "~/") {
		fullPath = filepath.Join(home, strings.TrimPrefix(fullPath, "~"))
	}

	if err := os.Remove(fullPath); err != nil && !os.IsNotExist(err) {
		json.NewEncoder(w).Encode(map[string]string{"error": "failed to remove file: " + err.Error()})
		return
	}

	if err := db.RemoveAgentConfig(req.AgentType, installPath); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "failed to remove config: " + err.Error()})
		return
	}

	json.NewEncoder(w).Encode(map[string]string{"status": "uninstalled", "path": fullPath})
}

func generateAgentInstructions(agentType string) string {
	prompts := mcp.GetPrompts()
	var promptText string
	for _, p := range prompts {
		if p.Name == "efficient-context-usage" {
			promptText = p.Prompt
			break
		}
	}

	switch agentType {
	case "claude_code":
		return "# Code Context Instructions\n\n" + promptText + "\n\nUse these tools for efficient code search:\n- get_context_capsule: Search code with token-efficient modes\n- cache_summary: Cache your own summaries\n- analyze_dead_code: Find unused code\n"
	case "cursor":
		return `{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}`
	case "opencode":
		return `{
  "mcpServers": {
    "ast-context-cache": {
      "url": "http://localhost:7821/mcp"
    }
  }
}`
	case "claude_desktop":
		return `{
  "mcpServers": {
    "ast-context-cache": {
      "command": "http",
      "url": "http://localhost:7821/mcp"
    }
  }
}`
	default:
		return promptText
	}
}

func handleResetProject(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
		return
	}
	var req map[string]string
	json.NewDecoder(r.Body).Decode(&req)
	projectPath := req["project_path"]

	if projectPath == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "project_path required"})
		return
	}
	projectPath = watcher.NormalizeProjectPath(projectPath)
	embedqueue.RemoveProject(projectPath)

	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")
	db.DB.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM vectors WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM queries WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM indexed_files WHERE project_path = ?", projectPath)
	db.DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	db.EnsureFTSTriggers()
	cache.GlobalCache.ClearProject(projectPath)
	go db.Compact()

	json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "project_path": projectPath})
}

func parseProjectPathFromRequest(r *http.Request) string {
	ct := r.Header.Get("Content-Type")
	if strings.Contains(ct, "application/json") {
		var req struct {
			ProjectPath string `json:"project_path"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return ""
		}
		return watcher.NormalizeProjectPath(req.ProjectPath)
	}
	r.ParseForm()
	return watcher.NormalizeProjectPath(r.FormValue("project_path"))
}

func respondHTMXPartial(w http.ResponseWriter, r *http.Request) {
	target := r.Header.Get("HX-Target")
	if strings.Contains(target, "settings-content") {
		handleSettingsPartial(w, r)
		return
	}
	if strings.Contains(target, "memory-panel") {
		handleMemoryPartial(w, r)
		return
	}
	handleIndexHealthPartial(w, r)
}

func handleStopWatcher(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	projectPath := parseProjectPathFromRequest(r)
	if projectPath != "" {
		_ = watcher.StopWatcher(projectPath)
	}
	respondHTMXPartial(w, r)
}

func handleStartWatcher(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	projectPath := parseProjectPathFromRequest(r)
	if projectPath != "" {
		watcher.EnsureWatcher(projectPath)
	}
	respondHTMXPartial(w, r)
}

func handleIndexProject(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	projectPath := parseProjectPathFromRequest(r)
	if projectPath == "" {
		http.Error(w, "project_path required", http.StatusBadRequest)
		return
	}
	if info, err := os.Stat(projectPath); err != nil {
		http.Error(w, "path not found: "+err.Error(), http.StatusBadRequest)
		return
	} else if !info.IsDir() {
		http.Error(w, "project_path must be a directory", http.StatusBadRequest)
		return
	}
	n, err := indexer.IndexDirectory(projectPath, projectPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	watcher.EnsureWatcher(projectPath)
	if mcp.GetEmbedder() != nil {
		go embedqueue.EnqueueAllSymbolsFiles(projectPath)
	}
	realtime.Notify(realtime.IndexCommitted | realtime.IndexHealth)
	w.Header().Set("X-Indexed-Symbols", strconv.Itoa(n))
	respondHTMXPartial(w, r)
}

func deleteProjectData(projectPath string) {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	embedqueue.RemoveProject(projectPath)
	watcher.DeleteWatcher(projectPath)
	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	db.DB.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")
	db.DB.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM vectors WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM queries WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM summaries WHERE project_path = ?", projectPath)
	db.DB.Exec("DELETE FROM indexed_files WHERE project_path = ?", projectPath)
	db.DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	db.EnsureFTSTriggers()
	cache.GlobalCache.ClearProject(projectPath)
	go db.Compact()
}

func handleDeleteWatcher(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	projectPath := parseProjectPathFromRequest(r)
	if projectPath == "" {
		respondHTMXPartial(w, r)
		return
	}
	deleteProjectData(projectPath)
	realtime.Notify(realtime.EmbedFinished | realtime.IndexHealth)
	respondHTMXPartial(w, r)
}

func handleSystemResources(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	allocMB := float64(memStats.Alloc) / (1024 * 1024)
	totalAllocMB := float64(memStats.TotalAlloc) / (1024 * 1024)
	sysMB := float64(memStats.Sys) / (1024 * 1024)

	dbPath := db.GetDBPath()
	var diskSize int64
	if fi, err := os.Stat(dbPath); err == nil {
		diskSize = fi.Size()
	}

	vectorMemMB := search.Cache.MemoryMB()
	queryCacheSize, queryCacheEntries := cache.GlobalCache.Stats()
	cacheHitRatio := cache.GlobalCache.HitRatio()
	diskIO := sys.DiskIORates()
	ssd := sys.SSDHealthInfo()
	load := sys.HostLoadAverage()

	json.NewEncoder(w).Encode(map[string]interface{}{
		"memory": map[string]float64{
			"alloc_mb":        allocMB,
			"total_alloc_mb":  totalAllocMB,
			"sys_mb":          sysMB,
			"vector_cache_mb": vectorMemMB,
		},
		"cpu_percent": sys.ProcessCPUPercent(),
		"load_avg": map[string]interface{}{
			"available":      load.Available,
			"load_1":           load.Load1,
			"load_5":           load.Load5,
			"load_15":          load.Load15,
			"cpus":             runtime.NumCPU(),
			"per_core_1":       load.Load1 / float64(max(runtime.NumCPU(), 1)),
			"per_core_5":       load.Load5 / float64(max(runtime.NumCPU(), 1)),
			"per_core_15":      load.Load15 / float64(max(runtime.NumCPU(), 1)),
			"utilization_1_pct": load.Load1 / float64(max(runtime.NumCPU(), 1)) * 100,
		},
		"disk": map[string]interface{}{
			"db_size_bytes":   diskSize,
			"wal_size_bytes":  db.WalFileBytes(),
			"read_mbps":       diskIO.ReadMBps,
			"write_mbps":      diskIO.WriteMBps,
			"ssd_model":       ssd.Model,
			"ssd_smart":       ssd.SmartStatus,
			"ssd_protocol":    ssd.Protocol,
			"ssd_capacity":    ssd.Capacity,
			"ssd_trim":        ssd.TrimSupport,
			"ssd_solid_state": ssd.SolidState,
			"ssd_wear_used_pct": ssd.WearUsedPct,
			"ssd_spare_pct":     ssd.SparePct,
			"ssd_data_written_tb": ssd.DataWrittenTB,
			"ssd_temperature_c":   ssd.TemperatureC,
		},
		"cache": map[string]interface{}{
			"query_cache_size":    queryCacheSize,
			"query_cache_entries": queryCacheEntries,
			"hit_ratio":        cacheHitRatio,
		},
		"goroutines": runtime.NumGoroutine(),
	})
}

func handleDocSources(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var req struct {
			Action string `json:"action"`
			ID     int    `json:"id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		if req.ID <= 0 {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"error": "id required"})
			return
		}
		switch req.Action {
		case "refresh", "update":
			docs.ForceRefreshSource(req.ID)
		case "delete":
			if err := docs.RemoveSource(req.ID); err != nil {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
				return
			}
		default:
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"error": "action refresh, update, or delete required"})
			return
		}
		if r.Header.Get("HX-Request") != "" {
			respondHTMXPartial(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "refreshing"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	sources, err := docs.ListSources()
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sources": sources,
		"total":   len(sources),
	})
}
