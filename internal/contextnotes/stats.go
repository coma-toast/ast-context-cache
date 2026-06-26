package contextnotes

import (
	"fmt"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// SessionRollup is per-session virtual context totals.
type SessionRollup struct {
	NotesCount            int `json:"session_notes_count"`
	VirtualTokensStored   int `json:"session_virtual_total"`
	VirtualTokensAccessed int `json:"session_virtual_accessed_total"`
}

// QuotaStrings for agent-visible quota display.
type QuotaStrings struct {
	SessionNotes  string `json:"notes"`
	SessionTokens string `json:"tokens"`
	GlobalNotes   string `json:"notes"`
	GlobalTokens  string `json:"tokens"`
}

// Inventory is live stored virtual context.
type Inventory struct {
	ActiveNotesCount   int
	ActiveInventoryTok int
	OrphanNotesCount   int
}

// DashboardStats aggregates virtual context metrics for the dashboard API.
type DashboardStats struct {
	ActiveInventoryTokens    int                    `json:"active_inventory_tokens"`
	ActiveNotesCount         int                    `json:"active_notes_count"`
	VirtualTokensStored30d   int                    `json:"virtual_tokens_stored_30d"`
	VirtualTokensAccessed30d int                    `json:"virtual_tokens_accessed_30d"`
	UtilizationPct30d        float64                `json:"utilization_pct_30d"`
	OrphanNotesCount         int                    `json:"orphan_notes_count"`
	TodayStored              int                    `json:"today_stored"`
	TodayAccessed            int                    `json:"today_accessed"`
	FlushedTokens30d         int                    `json:"flushed_tokens_30d"`
	Limits                   map[string]interface{} `json:"limits"`
	ByTool30d                map[string]ToolWindow          `json:"by_tool_30d"`
	KvRepair                 KvRepairDashboardStats         `json:"kv_repair"`
}

type ToolWindow struct {
	Calls         int `json:"calls"`
	VirtualTokens int `json:"virtual_tokens"`
}

func LiveInventory(projectPath string) Inventory {
	var inv Inventory
	where := "1=1"
	args := []any{}
	if projectPath != "" {
		where = "project_path = ?"
		args = append(args, projectPath)
	}
	db.ContextDB.QueryRow("SELECT COUNT(*), COALESCE(SUM(token_est),0), COALESCE(SUM(CASE WHEN access_count=0 THEN 1 ELSE 0 END),0) FROM context_notes WHERE "+where, args...).
		Scan(&inv.ActiveNotesCount, &inv.ActiveInventoryTok, &inv.OrphanNotesCount)
	return inv
}

func SessionRollupFor(sessionID string) SessionRollup {
	var r SessionRollup
	if sessionID == "" {
		return r
	}
	db.DB.QueryRow(`SELECT COALESCE(notes_count,0), COALESCE(virtual_tokens_stored,0), COALESCE(virtual_tokens_accessed,0)
		FROM context_session_stats WHERE session_id = ?`, sessionID).
		Scan(&r.NotesCount, &r.VirtualTokensStored, &r.VirtualTokensAccessed)
	if r.NotesCount == 0 {
		db.ContextDB.QueryRow(`SELECT COUNT(*), COALESCE(SUM(token_est),0) FROM context_notes WHERE session_id = ?`, sessionID).
			Scan(&r.NotesCount, &r.VirtualTokensStored)
	}
	return r
}

func GlobalRollup() (notes, tokens int) {
	db.ContextDB.QueryRow(`SELECT COUNT(*), COALESCE(SUM(token_est),0) FROM context_notes`).Scan(&notes, &tokens)
	return notes, tokens
}

func QuotaForSession(sessionID string) (sessionNotes, sessionTokens, globalNotes, globalTokens int) {
	if sessionID != "" {
		db.ContextDB.QueryRow(`SELECT COUNT(*), COALESCE(SUM(token_est),0) FROM context_notes WHERE session_id = ?`, sessionID).
			Scan(&sessionNotes, &sessionTokens)
	}
	globalNotes, globalTokens = GlobalRollup()
	return sessionNotes, sessionTokens, globalNotes, globalTokens
}

func BuildQuotaStats(sessionID string, lim Limits) map[string]interface{} {
	sn, st, gn, gt := QuotaForSession(sessionID)
	return map[string]interface{}{
		"session_quota": map[string]string{
			"notes":  fmt.Sprintf("%d/%d", sn, lim.MaxNotesSession),
			"tokens": fmt.Sprintf("%d/%d", st, lim.MaxTokensSession),
		},
		"global_quota": map[string]string{
			"notes":  fmt.Sprintf("%d/%d", gn, lim.MaxNotesGlobal),
			"tokens": fmt.Sprintf("%d/%d", gt, lim.MaxTokensGlobal),
		},
	}
}

func BuildStatsBlock(sessionID string, lim Limits) map[string]interface{} {
	r := SessionRollupFor(sessionID)
	stats := map[string]interface{}{
		"session_virtual_total":          r.VirtualTokensStored,
		"session_notes_count":            r.NotesCount,
		"session_virtual_accessed_total": r.VirtualTokensAccessed,
	}
	for k, v := range BuildQuotaStats(sessionID, lim) {
		stats[k] = v
	}
	inv := LiveInventory("")
	stats["active_inventory_tokens"] = inv.ActiveInventoryTok
	stats["active_notes_count"] = inv.ActiveNotesCount
	return stats
}

func bumpSessionStore(sessionID, projectPath string, tokenEst int) {
	now := time.Now().UTC().Format(time.RFC3339)
	db.DB.Exec(`INSERT INTO context_session_stats (session_id, project_path, notes_count, virtual_tokens_stored, last_store_at)
		VALUES (?, ?, 1, ?, ?)
		ON CONFLICT(session_id) DO UPDATE SET
			project_path = COALESCE(excluded.project_path, context_session_stats.project_path),
			notes_count = context_session_stats.notes_count + 1,
			virtual_tokens_stored = context_session_stats.virtual_tokens_stored + excluded.virtual_tokens_stored,
			last_store_at = excluded.last_store_at`, sessionID, projectPath, tokenEst, now)
}

func adjustSessionStore(sessionID string, notesDelta, tokensDelta int) {
	if sessionID == "" || (notesDelta == 0 && tokensDelta == 0) {
		return
	}
	db.DB.Exec(`UPDATE context_session_stats SET
		notes_count = CASE WHEN notes_count + ? < 0 THEN 0 ELSE notes_count + ? END,
		virtual_tokens_stored = CASE WHEN virtual_tokens_stored + ? < 0 THEN 0 ELSE virtual_tokens_stored + ? END
		WHERE session_id = ?`, notesDelta, notesDelta, tokensDelta, tokensDelta, sessionID)
}

func bumpSessionAccess(sessionID string, tokens int) {
	if sessionID == "" {
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	db.DB.Exec(`INSERT INTO context_session_stats (session_id, virtual_tokens_accessed, last_access_at)
		VALUES (?, ?, ?)
		ON CONFLICT(session_id) DO UPDATE SET
			virtual_tokens_accessed = context_session_stats.virtual_tokens_accessed + excluded.virtual_tokens_accessed,
			last_access_at = excluded.last_access_at`, sessionID, tokens, now)
}

func RecordAccess(ref, sessionID, projectPath, toolName string, virtualTokens int, repairReason string) {
	if ref == "" || virtualTokens <= 0 {
		return
	}
	reason := normalizeRepairReason(repairReason)
	now := time.Now().UTC().Format(time.RFC3339)
	db.ContextDB.Exec(`UPDATE context_notes SET access_count = access_count + 1,
		tokens_fetched = tokens_fetched + ?,
		last_accessed_at = ?
		WHERE ref = ?`, virtualTokens, now, ref)
	db.DB.Exec(`INSERT INTO context_note_access (ref, session_id, project_path, tool_name, virtual_tokens, repair_reason, accessed_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`, ref, sessionID, projectPath, toolName, virtualTokens, reason, now)
	bumpSessionAccess(sessionID, virtualTokens)
}

func DashboardStatsFor(projectPath string, windowDays int) DashboardStats {
	if windowDays <= 0 {
		windowDays = 30
	}
	lim := LoadLimits()
	inv := LiveInventory(projectPath)
	ds := DashboardStats{
		ActiveInventoryTokens: inv.ActiveInventoryTok,
		ActiveNotesCount:      inv.ActiveNotesCount,
		OrphanNotesCount:      inv.OrphanNotesCount,
		Limits:                lim.AsMap(),
		ByTool30d:             map[string]ToolWindow{},
	}
	cutoff := time.Now().AddDate(0, 0, -windowDays).Format("2006-01-02") + "T00:00:00"
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	toolFilter := "tool_name IN ('store_context','fetch_context','search_context','flush_context')"
	projectClause := ""
	args30 := []any{cutoff}
	argsToday := []any{todayStart, tomorrowStart}
	if projectPath != "" {
		projectClause = " AND project_path = ?"
		args30 = append(args30, projectPath)
		argsToday = append(argsToday, projectPath)
	}
	db.DB.QueryRow(`SELECT COALESCE(SUM(CASE WHEN tool_name='store_context' THEN tokens_saved ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN tool_name IN ('fetch_context','search_context') THEN tokens_used ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN tool_name='flush_context' THEN file_baseline_tokens ELSE 0 END),0)
		FROM queries WHERE timestamp >= ? AND `+toolFilter+projectClause, args30...).
		Scan(&ds.VirtualTokensStored30d, &ds.VirtualTokensAccessed30d, &ds.FlushedTokens30d)
	db.DB.QueryRow(`SELECT COALESCE(SUM(CASE WHEN tool_name='store_context' THEN tokens_saved ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN tool_name IN ('fetch_context','search_context') THEN tokens_used ELSE 0 END),0)
		FROM queries WHERE timestamp >= ? AND timestamp < ? AND `+toolFilter+projectClause, argsToday...).
		Scan(&ds.TodayStored, &ds.TodayAccessed)
	if ds.VirtualTokensStored30d > 0 {
		ds.UtilizationPct30d = float64(ds.VirtualTokensAccessed30d) / float64(ds.VirtualTokensStored30d) * 100
	}
	for _, tool := range []string{"store_context", "fetch_context", "search_context", "flush_context"} {
		var calls, tok int
		tokCol := "tokens_saved"
		if tool == "fetch_context" || tool == "search_context" {
			tokCol = "tokens_used"
		} else if tool == "flush_context" {
			tokCol = "file_baseline_tokens"
		}
		qargs := append([]any{cutoff, tool}, args30[1:]...)
		db.DB.QueryRow(`SELECT COUNT(*), COALESCE(SUM(`+tokCol+`),0) FROM queries
			WHERE timestamp >= ? AND tool_name = ?`+projectClause, qargs...).
			Scan(&calls, &tok)
		ds.ByTool30d[tool] = ToolWindow{Calls: calls, VirtualTokens: tok}
	}
	ds.KvRepair = KvRepairDashboardStatsFor(projectPath, windowDays)
	return ds
}
