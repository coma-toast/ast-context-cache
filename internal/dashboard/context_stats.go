package dashboard

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

func fillVirtualContextStats(s *components.Stats, projectPath string) {
	ds := contextnotes.DashboardStatsFor(projectPath, StatsWindowDays)
	s.VirtualInventoryTokens = ds.ActiveInventoryTokens
	s.VirtualNotesCount = ds.ActiveNotesCount
	s.VirtualUtilPct30d = ds.UtilizationPct30d
	s.VirtualOrphanCount = ds.OrphanNotesCount
	s.VirtualFlushed30d = ds.FlushedTokens30d
	s.VirtualStored30d = ds.VirtualTokensStored30d
	s.VirtualAccessed30d = ds.VirtualTokensAccessed30d
	s.VirtualTodayStored = ds.TodayStored
	s.VirtualTodayAccessed = ds.TodayAccessed
	fillKvRepairStats(s, ds.KvRepair)
	if ds.Limits != nil {
		if v, ok := ds.Limits["max_notes_global"].(int); ok {
			s.VirtualMaxNotesGlobal = v
		}
		if v, ok := ds.Limits["max_tokens_global"].(int); ok {
			s.VirtualMaxTokensGlobal = v
		}
	}
}

func fillKvRepairStats(s *components.Stats, kv contextnotes.KvRepairDashboardStats) {
	s.KvRepairArchivesActive = kv.ArchivesActive
	s.KvRepairArchivesStored30d = kv.ArchivesStored30d
	s.KvRepairRepairsTotal30d = kv.RepairsTotal30d
	s.KvRepairUtilPct30d = kv.RepairUtilizationPct30d
	s.KvRepairOrphans = kv.RepairOrphans
	s.KvRepairTokensRepaired30d = kv.TokensRepaired30d
	s.KvRepairCacheMiss30d = kv.CacheMissSignals30d
	s.KvRepairQuality30d = kv.QualitySignals30d
	s.KvRepairManual30d = kv.ManualSignals30d
	s.KvRepairTodayRepairs = kv.TodayRepairs
}

func handleKvRepairStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	kv := contextnotes.KvRepairDashboardStatsFor(pid, StatsWindowDays)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(kv)
}

func handleContextStats(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	ds := contextnotes.DashboardStatsFor(pid, StatsWindowDays)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ds)
}

func handleFlushContextAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var body map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	sessionID, _ := body["session_id"].(string)
	projectPath, _ := body["project_path"].(string)
	all, _ := body["all"].(bool)
	refs := body["refs"]
	if refs == nil {
		if r, ok := body["ref"].(string); ok {
			refs = r
		}
	}
	res, err := contextnotes.Flush(sessionID, refs, projectPath, all)
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	realtime.Notify(realtime.Stats | realtime.IndexHealth)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"flushed_refs":         res.FlushedRefs,
		"virtual_tokens_freed": res.VirtualTokensFreed,
		"scope":                res.Scope,
		"stats":                res.Stats,
	})
}

func populateContextSettings(settings map[string]string, data *components.SettingsData) {
	data.ContextMaxNotesSession = intSetting(settings, "context_max_notes_session", 50)
	data.ContextMaxTokensSession = intSetting(settings, "context_max_tokens_session", 32000)
	data.ContextMaxNotesGlobal = intSetting(settings, "context_max_notes_global", 500)
	data.ContextMaxTokensGlobal = intSetting(settings, "context_max_tokens_global", 200000)
	data.ContextLimitPolicy = settings["context_limit_policy"]
	if data.ContextLimitPolicy == "" {
		data.ContextLimitPolicy = "reject"
	}
}

func intSetting(settings map[string]string, key string, def int) int {
	v := settings[key]
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return def
	}
	return n
}
