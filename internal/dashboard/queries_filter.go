package dashboard

// SQL fragments shared by the dashboard's queries-table aggregates.
const (
	// file_watcher is logged for fsnotify indexing, not MCP tool usage.
	excludeWatcherFromToolStats = "tool_name != 'file_watcher'"
	onlyWatcherFilter           = "tool_name = 'file_watcher'"
	queriesRollingWindow        = "timestamp >= datetime('now', '-30 days')"
	projectPathClause           = " AND project_path = ?"
	// savingsToolsClause limits "Tokens saved" to compression + dedup from search/read tools;
	// virtual context and memory writes (store_context, store_memory, ...) are not savings.
	savingsToolsClause = "tool_name IN ('get_context_capsule','search_semantic','get_file_context','retrieve','execute_code')"
	tokensSavedSum     = "COALESCE(SUM(CASE WHEN " + savingsToolsClause + " THEN tokens_saved ELSE 0 END),0)"
	dedupTokensSum     = "COALESCE(SUM(CASE WHEN " + savingsToolsClause + " THEN dedup_tokens_saved ELSE 0 END),0)"
	savingsVsFilesSum  = "COALESCE(SUM(CASE WHEN " + savingsToolsClause + " THEN savings_vs_files ELSE 0 END),0)"
	// tokensSavedCol / dedupTokensSavedCol are the per-row equivalents for row listings.
	tokensSavedCol      = "CASE WHEN " + savingsToolsClause + " THEN COALESCE(tokens_saved,0) ELSE 0 END"
	dedupTokensSavedCol = "CASE WHEN " + savingsToolsClause + " THEN COALESCE(dedup_tokens_saved,0) ELSE 0 END"
	// compressionLedgerClause matches rows on the compression ledger; rows logged before
	// queries.ledger existed (ledger '') fall back to the savings tools.
	compressionLedgerClause = "(ledger = 'compression' OR (COALESCE(ledger,'') = '' AND " + savingsToolsClause + "))"
	compressionSavedSum     = "COALESCE(SUM(CASE WHEN " + compressionLedgerClause + " THEN MAX(0, COALESCE(tokens_saved,0) - COALESCE(dedup_tokens_saved,0)) ELSE 0 END),0)"
	ledgerDedupSavedSum     = "COALESCE(SUM(CASE WHEN " + compressionLedgerClause + " THEN COALESCE(dedup_tokens_saved,0) ELSE 0 END),0)"
	conservativeSavedSum    = "COALESCE(SUM(CASE WHEN " + compressionLedgerClause + " AND COALESCE(conservative_baseline_tokens,0) > 0 THEN MAX(0, conservative_baseline_tokens - COALESCE(tokens_used,0)) ELSE 0 END),0)"
	// The virtual ledger: tokens written to stored context/memory, and tokens later
	// fetched or recalled from it.
	virtualStoredSum   = "COALESCE(SUM(CASE WHEN tool_name IN ('store_context','store_memory') THEN COALESCE(tokens_saved,0) ELSE 0 END),0)"
	virtualFetchedSum  = "COALESCE(SUM(CASE WHEN tool_name = 'fetch_context' THEN COALESCE(tokens_used,0) ELSE 0 END),0)"
	virtualRecalledSum = "COALESCE(SUM(CASE WHEN tool_name = 'recall_memory' THEN COALESCE(tokens_used,0) ELSE 0 END),0)"
	estimatedRowsSum   = "COALESCE(SUM(CASE WHEN COALESCE(estimate_method,'bytes4') = 'bytes4' THEN 1 ELSE 0 END),0)"
)

// StatsWindowDays is the rolling window for dashboard aggregate totals.
const StatsWindowDays = 30

func statsQueriesWhere(projectID string) (where string, args []any) {
	where = queriesRollingWindow
	if projectID != "" {
		where += projectPathClause
		args = append(args, projectID)
	}
	return
}

func toolStatsWhere(projectID string) (where string, args []any) {
	where = excludeWatcherFromToolStats + " AND " + queriesRollingWindow
	if projectID != "" {
		where += projectPathClause
		args = append(args, projectID)
	}
	return
}
