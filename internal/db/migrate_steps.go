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
)

var indexSteps []schemaStep

var contextSteps = []schemaStep{
	{version: 1, name: "normalize_timestamps", run: execSteps(normalizeContextNotesLastAccessedQuery)},
}

var usageSteps = []schemaStep{
	{version: 1, name: "normalize_timestamps", run: execSteps(
		normalizeContextNoteAccessAtQuery, normalizeSessionStatsLastStoreQuery, normalizeSessionStatsLastAccessQuery,
	)},
	{version: 2, name: "queries_estimate_method", run: execSteps(addQueriesEstimateMethodColumn)},
}
