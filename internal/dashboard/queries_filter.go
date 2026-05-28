package dashboard

// file_watcher is logged for fsnotify indexing, not MCP tool usage.
const excludeWatcherFromToolStats = "tool_name != 'file_watcher'"

// StatsWindowDays is the rolling window for dashboard aggregate totals.
const StatsWindowDays = 30

const queriesRollingWindow = "timestamp >= datetime('now', '-30 days')"

const (
	tokensSavedSum    = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN tokens_saved ELSE 0 END),0)"
	dedupTokensSum    = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN dedup_tokens_saved ELSE 0 END),0)"
	savingsVsFilesSum = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN savings_vs_files ELSE 0 END),0)"
)

func statsQueriesWhere(projectID string) (where string, args []any) {
	where = queriesRollingWindow
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	return
}

func toolStatsWhere(projectID string) (where string, args []any) {
	where = excludeWatcherFromToolStats + " AND " + queriesRollingWindow
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	return
}
