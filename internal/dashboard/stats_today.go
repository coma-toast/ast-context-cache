package dashboard

import (
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	selectTodayStatsQuery = "SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(AVG(duration_ms),0), " + tokensSavedSum + " FROM queries WHERE timestamp >= ? AND timestamp < ?"
)

func fillTodayStats(pid, todayStart, tomorrowStart string, s *components.Stats) {
	if db.DB == nil {
		return
	}
	todaySel := selectTodayStatsQuery
	args := []any{todayStart, tomorrowStart}
	if pid != "" {
		todaySel += projectPathClause
		args = append(args, pid)
	}
	db.DB.QueryRow(todaySel, args...).Scan(&s.TodayQueries, &s.TodaySessions, &s.TodayAvgDurationMs, &s.TodayTokens)
}
