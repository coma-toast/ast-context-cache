package dashboard

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
)

// tokensPerRoundEstimate is the naïve context budget used for rounds-avoided heuristic.
const tokensPerRoundEstimate = 4000

const (
	tokensUsedSum         = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN tokens_used ELSE 0 END),0)"
	symbolBaselineSumExpr = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN symbol_baseline_tokens ELSE 0 END),0)"
)

// ValueHeuristic is an approximate before/after estimate of agent token value.
type ValueHeuristic struct {
	ApproxBaselineTokens int     `json:"ApproxBaselineTokens"`
	ApproxTokensReturned int     `json:"ApproxTokensReturned"`
	ApproxTokensSaved    int     `json:"ApproxTokensSaved"`
	ApproxRoundsAvoided  float64 `json:"ApproxRoundsAvoided"`
	HeuristicApproximate bool    `json:"HeuristicApproximate"`
	HeuristicLabel       string  `json:"HeuristicLabel"`
	WindowDays           int     `json:"WindowDays"`
}

// WeeklyDigestTool is a top tool row in the weekly digest.
type WeeklyDigestTool struct {
	ToolName    string  `json:"ToolName"`
	Calls       int     `json:"Calls"`
	TokensSaved int     `json:"TokensSaved"`
	AvgDuration float64 `json:"AvgDurationMs"`
}

// WeeklyDigestEmbedReliability summarizes embed queue health for the digest card.
type WeeklyDigestEmbedReliability struct {
	PendingFailures     int64  `json:"PendingFailures"`
	LastAutoRecoverUnix int64  `json:"LastAutoRecoverUnix"`
	AbnormalPreviousRun bool   `json:"AbnormalPreviousRun"`
	Available           bool   `json:"Available"`
	Note                string `json:"Note,omitempty"`
}

// WeeklyDigest is a 7-day rollup for the Overview digest card.
type WeeklyDigest struct {
	WindowDays       int                          `json:"WindowDays"`
	TokensSaved      int                          `json:"TokensSaved"`
	Queries          int                          `json:"Queries"`
	VirtualStored    int                          `json:"VirtualStored"`
	VirtualAccessed  int                          `json:"VirtualAccessed"`
	TopTools         []WeeklyDigestTool           `json:"TopTools"`
	EmbedReliability WeeklyDigestEmbedReliability `json:"EmbedReliability"`
	Heuristic        ValueHeuristic               `json:"Heuristic"`
}

// ContextSessionStory is a compact per-session virtual-context summary.
type ContextSessionStory struct {
	SessionID             string `json:"SessionID"`
	ProjectPath           string `json:"ProjectPath,omitempty"`
	NotesCount            int    `json:"NotesCount"`
	VirtualTokensStored   int    `json:"VirtualTokensStored"`
	VirtualTokensAccessed int    `json:"VirtualTokensAccessed"`
	ActiveNotes           int    `json:"ActiveNotes"`
	ActiveTokens          int    `json:"ActiveTokens"`
	FetchedAfterStore     bool   `json:"FetchedAfterStore"`
	LastStoreAt           string `json:"LastStoreAt,omitempty"`
	LastAccessAt          string `json:"LastAccessAt,omitempty"`
}

// ContextSessionsResponse wraps session stories for Overview.
type ContextSessionsResponse struct {
	WindowDays int                   `json:"WindowDays"`
	Sessions   []ContextSessionStory `json:"Sessions"`
}

// statsWithHeuristic enriches dashboard stats JSON with approximate value metrics.
type statsWithHeuristic struct {
	ApproxBaselineTokens int     `json:"ApproxBaselineTokens"`
	ApproxTokensReturned int     `json:"ApproxTokensReturned"`
	ApproxRoundsAvoided  float64 `json:"ApproxRoundsAvoided"`
	HeuristicApproximate bool    `json:"HeuristicApproximate"`
	HeuristicLabel       string  `json:"HeuristicLabel"`
}

func computeValueHeuristic(tokensSaved, tokensReturned, symbolBaseline int, windowDays int) ValueHeuristic {
	baseline := symbolBaseline
	if baseline <= 0 {
		baseline = tokensSaved + tokensReturned
	}
	saved := tokensSaved
	if saved <= 0 && baseline > tokensReturned {
		saved = baseline - tokensReturned
	}
	rounds := 0.0
	if saved > 0 {
		rounds = float64(saved) / float64(tokensPerRoundEstimate)
	}
	if windowDays <= 0 {
		windowDays = StatsWindowDays
	}
	return ValueHeuristic{
		ApproxBaselineTokens: baseline,
		ApproxTokensReturned: tokensReturned,
		ApproxTokensSaved:    saved,
		ApproxRoundsAvoided:  rounds,
		HeuristicApproximate: true,
		HeuristicLabel:       "approximate",
		WindowDays:           windowDays,
	}
}

func queryTokensReturnedAndBaseline(projectID string, days int) (tokensReturned, symbolBaseline int) {
	if db.DB == nil {
		return 0, 0
	}
	where := "timestamp >= datetime('now', ?)"
	args := []any{fmtDaysOffset(days)}
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	db.DB.QueryRow(`SELECT `+tokensUsedSum+`, `+symbolBaselineSumExpr+` FROM queries WHERE `+where, args...).
		Scan(&tokensReturned, &symbolBaseline)
	return tokensReturned, symbolBaseline
}

func queryTokensSavedWindow(projectID string, days int) (tokensSaved, queries int) {
	if db.DB == nil {
		return 0, 0
	}
	where := "timestamp >= datetime('now', ?)"
	args := []any{fmtDaysOffset(days)}
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	db.DB.QueryRow(`SELECT COUNT(*), `+tokensSavedSum+` FROM queries WHERE `+where, args...).
		Scan(&queries, &tokensSaved)
	return tokensSaved, queries
}

func queryVirtualWindow(projectID string, days int) (stored, accessed int) {
	if db.DB == nil {
		return 0, 0
	}
	where := "timestamp >= datetime('now', ?) AND tool_name IN ('store_context','fetch_context','search_context','flush_context')"
	args := []any{fmtDaysOffset(days)}
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	db.DB.QueryRow(`SELECT
		COALESCE(SUM(CASE WHEN tool_name='store_context' THEN tokens_saved ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN tool_name IN ('fetch_context','search_context') THEN tokens_used ELSE 0 END),0)
		FROM queries WHERE `+where, args...).Scan(&stored, &accessed)
	return stored, accessed
}

func queryTopToolsWindow(projectID string, days, limit int) []WeeklyDigestTool {
	if db.DB == nil || limit <= 0 {
		return nil
	}
	where := "tool_name != 'file_watcher' AND timestamp >= datetime('now', ?)"
	args := []any{fmtDaysOffset(days)}
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	q := `SELECT tool_name, COUNT(*), COALESCE(SUM(tokens_saved),0), COALESCE(AVG(duration_ms),0)
		FROM queries WHERE ` + where + ` GROUP BY tool_name ORDER BY COUNT(*) DESC LIMIT ?`
	args = append(args, limit)
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []WeeklyDigestTool
	for rows.Next() {
		var t WeeklyDigestTool
		if err := rows.Scan(&t.ToolName, &t.Calls, &t.TokensSaved, &t.AvgDuration); err != nil {
			continue
		}
		out = append(out, t)
	}
	return out
}

func buildEmbedReliability() WeeklyDigestEmbedReliability {
	eq := embedqueue.Snapshot()
	r := WeeklyDigestEmbedReliability{
		PendingFailures:     eq.Failed,
		LastAutoRecoverUnix: eq.LastAutoRecoverUnix,
		AbnormalPreviousRun: embedqueue.AbnormalPreviousRun(),
		Available:           true,
		Note:                "embed failures are process-lifetime pending counts; auto-recover is last stuck-worker event",
	}
	return r
}

func buildWeeklyDigest(projectID string) WeeklyDigest {
	const days = 7
	saved, queries := queryTokensSavedWindow(projectID, days)
	stored, accessed := queryVirtualWindow(projectID, days)
	returned, baseline := queryTokensReturnedAndBaseline(projectID, days)
	return WeeklyDigest{
		WindowDays:       days,
		TokensSaved:      saved,
		Queries:          queries,
		VirtualStored:    stored,
		VirtualAccessed:  accessed,
		TopTools:         queryTopToolsWindow(projectID, days, 5),
		EmbedReliability: buildEmbedReliability(),
		Heuristic:        computeValueHeuristic(saved, returned, baseline, days),
	}
}

func buildContextSessions(projectID string, days, limit int) ContextSessionsResponse {
	if days <= 0 {
		days = StatsWindowDays
	}
	if limit <= 0 {
		limit = 20
	}
	resp := ContextSessionsResponse{WindowDays: days, Sessions: []ContextSessionStory{}}
	if db.DB == nil {
		return resp
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -days).Format(time.RFC3339)
	where := `(COALESCE(last_store_at,'') >= ? OR COALESCE(last_access_at,'') >= ?)`
	args := []any{cutoff, cutoff}
	if projectID != "" {
		where += ` AND project_path = ?`
		args = append(args, projectID)
	}
	q := `SELECT session_id, COALESCE(project_path,''), COALESCE(notes_count,0),
		COALESCE(virtual_tokens_stored,0), COALESCE(virtual_tokens_accessed,0),
		COALESCE(last_store_at,''), COALESCE(last_access_at,'')
		FROM context_session_stats WHERE ` + where + `
		ORDER BY CASE WHEN last_access_at > last_store_at THEN last_access_at ELSE last_store_at END DESC
		LIMIT ?`
	args = append(args, limit)
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return resp
	}
	defer rows.Close()
	for rows.Next() {
		var s ContextSessionStory
		if err := rows.Scan(&s.SessionID, &s.ProjectPath, &s.NotesCount, &s.VirtualTokensStored, &s.VirtualTokensAccessed, &s.LastStoreAt, &s.LastAccessAt); err != nil {
			continue
		}
		s.FetchedAfterStore = s.VirtualTokensAccessed > 0 && s.VirtualTokensStored > 0
		if db.ContextDB != nil && s.SessionID != "" {
			db.ContextDB.QueryRow(`SELECT COUNT(*), COALESCE(SUM(token_est),0) FROM context_notes WHERE session_id = ?`, s.SessionID).
				Scan(&s.ActiveNotes, &s.ActiveTokens)
		}
		resp.Sessions = append(resp.Sessions, s)
	}
	return resp
}

func fmtDaysOffset(days int) string {
	if days <= 0 {
		days = StatsWindowDays
	}
	return "-" + strconv.Itoa(days) + " days"
}

func handleDashboardWeeklyDigestJSON(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildWeeklyDigest(pid))
}

func handleDashboardContextSessionsJSON(w http.ResponseWriter, r *http.Request) {
	pid := r.URL.Query().Get("project_id")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildContextSessions(pid, StatsWindowDays, 20))
}
