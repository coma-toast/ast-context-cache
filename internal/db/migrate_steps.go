package db

// Schema steps, one list per database. Append new steps at the end with the next version.
// Steps must stay additive (new tables and columns only: no DROP, RENAME or column type
// change) so a 4.x binary still runs against a 5.x data directory; migrate_test.go
// enforces this over every string constant in this file.

const (
	// RFC3339 values ("…T…") written before timestamps were normalized become SQLite's
	// datetime() form, so they compare correctly as text with datetime('now') values. A
	// value datetime() cannot parse is left as is.
	normalizeContextNotesLastAccessedQuery = `UPDATE context_notes SET last_accessed_at = datetime(last_accessed_at)
		WHERE last_accessed_at LIKE '%T%' AND datetime(last_accessed_at) IS NOT NULL`
	normalizeContextNoteAccessAtQuery = `UPDATE context_note_access SET accessed_at = datetime(accessed_at)
		WHERE accessed_at LIKE '%T%' AND datetime(accessed_at) IS NOT NULL`
	normalizeSessionStatsLastStoreQuery = `UPDATE context_session_stats SET last_store_at = datetime(last_store_at)
		WHERE last_store_at LIKE '%T%' AND datetime(last_store_at) IS NOT NULL`
	normalizeSessionStatsLastAccessQuery = `UPDATE context_session_stats SET last_access_at = datetime(last_access_at)
		WHERE last_access_at LIKE '%T%' AND datetime(last_access_at) IS NOT NULL`
	addQueriesEstimateMethodColumn = `ALTER TABLE queries ADD COLUMN estimate_method TEXT DEFAULT 'bytes4'`
	// Offload notes leave a tombstone when they expire, so fetch_context can report the ref
	// as expired (with the tool and args that produced it) instead of silently missing it.
	createContextNoteTombstonesTable = `CREATE TABLE IF NOT EXISTS context_note_tombstones (
		ref TEXT PRIMARY KEY, kind TEXT, tool TEXT, args_json TEXT, expired_at TEXT)`
	addQueriesConservativeBaselineColumn = `ALTER TABLE queries ADD COLUMN conservative_baseline_tokens INTEGER DEFAULT 0`
	addQueriesLedgerColumn               = `ALTER TABLE queries ADD COLUMN ledger TEXT DEFAULT ''`
	// Transcript usage ingest (TL-5): per-file resume offsets and per-day usage totals.
	// Only token counts are stored, never transcript text.
	createHostUsageOffsetsTable = `CREATE TABLE IF NOT EXISTS host_usage_offsets (
		path TEXT PRIMARY KEY, offset INTEGER NOT NULL DEFAULT 0, mtime INTEGER NOT NULL DEFAULT 0)`
	createHostUsageDailyTable = `CREATE TABLE IF NOT EXISTS host_usage_daily (
		day TEXT NOT NULL, project_dir TEXT NOT NULL,
		input INTEGER NOT NULL DEFAULT 0, output INTEGER NOT NULL DEFAULT 0,
		cache_read INTEGER NOT NULL DEFAULT 0, cache_write INTEGER NOT NULL DEFAULT 0,
		PRIMARY KEY (day, project_dir))`
)

var indexSteps []schemaStep

var contextSteps = []schemaStep{
	{version: 1, name: "normalize_timestamps", run: execSteps(normalizeContextNotesLastAccessedQuery)},
	{version: 2, name: "context_note_tombstones", run: execSteps(createContextNoteTombstonesTable)},
	{version: 3, name: "recount_token_est", run: recountTokenEstimates},
}

var usageSteps = []schemaStep{
	{version: 1, name: "normalize_timestamps", run: execSteps(
		normalizeContextNoteAccessAtQuery, normalizeSessionStatsLastStoreQuery, normalizeSessionStatsLastAccessQuery,
	)},
	{version: 2, name: "queries_estimate_method", run: execSteps(addQueriesEstimateMethodColumn)},
	{version: 3, name: "queries_ledgers", run: execSteps(addQueriesConservativeBaselineColumn, addQueriesLedgerColumn)},
	{version: 4, name: "host_usage", run: execSteps(createHostUsageOffsetsTable, createHostUsageDailyTable)},
}
