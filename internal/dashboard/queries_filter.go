package dashboard

// file_watcher is logged for fsnotify indexing, not MCP tool usage.
const excludeWatcherFromToolStats = "tool_name != 'file_watcher'"

const (
	tokensSavedSum    = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN tokens_saved ELSE 0 END),0)"
	dedupTokensSum    = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN dedup_tokens_saved ELSE 0 END),0)"
	savingsVsFilesSum = "COALESCE(SUM(CASE WHEN tool_name != 'file_watcher' THEN savings_vs_files ELSE 0 END),0)"
)
