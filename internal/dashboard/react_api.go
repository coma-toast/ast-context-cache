package dashboard

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

func registerReactAPI(mux *http.ServeMux) {
	mux.HandleFunc("/api/dashboard/health", handleDashboardHealthJSON)
	mux.HandleFunc("/api/dashboard/stats", handleDashboardStatsJSON)
	mux.HandleFunc("/api/dashboard/weekly-digest", handleDashboardWeeklyDigestJSON)
	mux.HandleFunc("/api/dashboard/context-sessions", handleDashboardContextSessionsJSON)
	mux.HandleFunc("/api/dashboard/index-health", handleDashboardIndexHealthJSON)
	mux.HandleFunc("/api/dashboard/memory", handleDashboardMemoryJSON)
	mux.HandleFunc("/api/dashboard/settings", handleDashboardSettingsJSON)
	mux.HandleFunc("/api/dashboard/recent-split", handleDashboardRecentSplitJSON)
	mux.HandleFunc("/api/dashboard/recent-logs", handleDashboardRecentLogsJSON)
	mux.HandleFunc("/api/dashboard/mcp-tier", handleDashboardMCPTierJSON)
}

func handleDashboardRecentLogsJSON(w http.ResponseWriter, r *http.Request) {
	logs, path, truncated, opts := buildRecentLogsForDashboard()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"lines":          logs,
		"path":           path,
		"file_truncated": truncated,
		"tail_lines":     opts.TailLines,
	})
}

func handleDashboardHealthJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildHealthData())
}

func buildHealthData() components.Health {
	state, lastUse := mcp.EmbedderState()
	embedErr := mcp.EmbedderError()
	eq := embedqueue.Snapshot()
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	heapMB := float64(memStats.HeapAlloc) / (1024 * 1024)
	h := components.Health{
		EmbedderState:         state,
		EmbedderError:         embedErr,
		EmbedderLast:          lastUse,
		QueueWorkers:          eq.Workers,
		QueueWorkersEffective: eq.WorkersEffective,
		QueueWorkersLive:      eq.WorkersLive,
		QueueThroughput:       eq.Throughput,
		QueueQueued:           eq.Queued,
		QueuePending:          eq.Pending,
		QueuePendingPeak:      eq.PendingPeak,
		QueueInFlight:         eq.InFlight,
		QueueHighCap:          eq.HighCap,
		QueueLowCap:           eq.LowCap,
		CacheHitRatio:         cache.GlobalCache.HitRatio(),
		HeapMB:                heapMB,
		CPUPercent:            sys.ProcessCPUPercent(),
		Uptime:                time.Since(serverStartTime),
		Version:               version.Version,
		AbnormalPreviousRun:   embedqueue.AbnormalPreviousRun(),
	}
	applyActiveEmbedderHealth(&h)
	overlayStartupEmbedder(&h.EmbedderState, &h.EmbedderError, &h.EmbedBackend)
	return h
}

func handleDashboardStatsJSON(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	s := components.Stats{}
	h := ValueHeuristic{HeuristicApproximate: true, HeuristicLabel: "approximate", WindowDays: StatsWindowDays}
	if usageDBReady() {
		todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
		tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
		statsSel := "SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(result_chars),0), COALESCE(AVG(duration_ms),0), " + tokensSavedSum + ", " + dedupTokensSum + ", " + savingsVsFilesSum + " FROM queries WHERE "
		where, args := statsQueriesWhere(pid)
		db.DB.QueryRow(statsSel+where, args...).
			Scan(&s.TotalQueries, &s.Sessions, &s.TotalChars, &s.AvgDurationMs, &s.TokensSaved, &s.DedupTokensSaved, &s.SavingsVsFiles)
		fillTodayStats(pid, todayStart, tomorrowStart, &s)
		fillVirtualContextStats(&s, pid)
		returned, baseline := queryTokensReturnedAndBaseline(pid, StatsWindowDays)
		h = computeValueHeuristic(s.TokensSaved, returned, baseline, StatsWindowDays)
		s.SymbolBaseline = h.ApproxBaselineTokens
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(struct {
		components.Stats
		statsWithHeuristic
	}{
		Stats: s,
		statsWithHeuristic: statsWithHeuristic{
			ApproxBaselineTokens: h.ApproxBaselineTokens,
			ApproxTokensReturned: h.ApproxTokensReturned,
			ApproxRoundsAvoided:  h.ApproxRoundsAvoided,
			HeuristicApproximate: h.HeuristicApproximate,
			HeuristicLabel:       h.HeuristicLabel,
		},
	})
}

func handleDashboardIndexHealthJSON(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildIndexHealth(pid))
}

func handleDashboardMemoryJSON(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	page := parseDocSourcesPageQuery(r)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildMemory(pid, page))
}

func handleDashboardSettingsJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildSettingsData(settingsBuildOpts{loadEmbedModels: true}))
}

func handleDashboardRecentSplitJSON(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	mcpQ, idxQ := buildRecentQueries(pid, 50)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"mcp":      mcpQ,
		"indexing": idxQ,
	})
}

func handleDashboardMCPTierJSON(w http.ResponseWriter, r *http.Request) {
	tier := os.Getenv("AST_MCP_TIER")
	if tier == "" {
		tier = "extended"
	}
	toolsPath := filepath.Join(os.Getenv("HOME"), ".astcache", "tools.json")
	if home, err := os.UserHomeDir(); err == nil {
		toolsPath = filepath.Join(home, ".astcache", "tools.json")
	}
	var toolsExists bool
	if _, err := os.Stat(toolsPath); err == nil {
		toolsExists = true
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tier":              tier,
		"tools_json_path":   toolsPath,
		"tools_json_exists": toolsExists,
	})
}
