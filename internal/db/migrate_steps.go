package db

import "database/sql"

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
	// MEM-1/MEM-4: invalidated rows keep a reason (contradiction, user_forget, ttl,
	// stale_anchor, quarantine, restored) so history output can explain the chain.
	addStructuredMemoryInvalidatedReasonColumn = `ALTER TABLE structured_memory ADD COLUMN invalidated_reason TEXT`
	// MEM-5: world time — when a fact is true in the world, not when the system learned it.
	addStructuredMemoryWorldValidAtColumn   = `ALTER TABLE structured_memory ADD COLUMN world_valid_at TEXT`
	addStructuredMemoryWorldInvalidAtColumn = `ALTER TABLE structured_memory ADD COLUMN world_invalid_at TEXT`
	// MEM-8: TTL expiry. The hourly memory expiry job and PruneArchived read this column.
	addStructuredMemoryExpiresAtColumn = `ALTER TABLE structured_memory ADD COLUMN expires_at TEXT`
	createStructuredMemoryExpiresIndex = `CREATE INDEX IF NOT EXISTS idx_struct_mem_expires ON structured_memory(expires_at)`
	// MEM-1/TR-3: quarantine state. A quarantined entry is excluded from default recall/search
	// and invalidated with reason quarantine when rejected (RejectQuarantined).
	addStructuredMemoryQuarantinedColumn = `ALTER TABLE structured_memory ADD COLUMN quarantined INTEGER DEFAULT 0`
	// KB: knowledge-base / audit columns, indexes, and FTS5 index on structured_memory.
	addStructuredMemoryTitleColumn          = `ALTER TABLE structured_memory ADD COLUMN title TEXT`
	addStructuredMemoryBodyColumn           = `ALTER TABLE structured_memory ADD COLUMN body TEXT`
	addStructuredMemoryTagsColumn           = `ALTER TABLE structured_memory ADD COLUMN tags TEXT`
	addStructuredMemoryActorColumn          = `ALTER TABLE structured_memory ADD COLUMN actor TEXT`
	addStructuredMemoryCategoryColumn       = `ALTER TABLE structured_memory ADD COLUMN category TEXT`
	addStructuredMemorySeverityColumn       = `ALTER TABLE structured_memory ADD COLUMN severity TEXT`
	addStructuredMemoryTargetColumn         = `ALTER TABLE structured_memory ADD COLUMN target TEXT`
	addStructuredMemoryBeforeValueColumn    = `ALTER TABLE structured_memory ADD COLUMN before_value TEXT`
	addStructuredMemoryAfterValueColumn     = `ALTER TABLE structured_memory ADD COLUMN after_value TEXT`
	addStructuredMemoryReasonColumn         = `ALTER TABLE structured_memory ADD COLUMN reason TEXT`
	addStructuredMemoryOccurredAtColumn     = `ALTER TABLE structured_memory ADD COLUMN occurred_at TEXT`
	addStructuredMemoryVerifiedColumn       = `ALTER TABLE structured_memory ADD COLUMN verified INTEGER DEFAULT 0`
	addStructuredMemoryVerifiedAtColumn     = `ALTER TABLE structured_memory ADD COLUMN verified_at TEXT`
	addStructuredMemoryRedactedFieldsColumn = `ALTER TABLE structured_memory ADD COLUMN redacted_fields TEXT`
	createStructuredMemoryKindOccurredIndex = `CREATE INDEX IF NOT EXISTS idx_struct_mem_kind_occurred ON structured_memory(kind, occurred_at)`
	createStructuredMemoryTargetIndex       = `CREATE INDEX IF NOT EXISTS idx_struct_mem_target ON structured_memory(kind, target, scope, valid_until)`
	createStructuredMemoryKBFTSStepTable    = `CREATE VIRTUAL TABLE IF NOT EXISTS structured_memory_kb_fts USING fts5(ref, title, subject, predicate, object, rule, body, tags, target, before_value, after_value, reason, category, actor)`
	backfillStructuredMemoryKBFTSQuery      = `INSERT INTO structured_memory_kb_fts (ref, title, subject, predicate, object, rule, body, tags, target, before_value, after_value, reason, category, actor)
		SELECT COALESCE(ref, ''), COALESCE(title, ''), COALESCE(subject, ''), COALESCE(predicate, ''), COALESCE(object, ''), COALESCE(rule, ''), COALESCE(body, ''), COALESCE(tags, ''), COALESCE(target, ''), COALESCE(before_value, ''), COALESCE(after_value, ''), COALESCE(reason, ''), COALESCE(category, ''), COALESCE(actor, '')
		FROM structured_memory
		WHERE ref NOT IN (SELECT ref FROM structured_memory_kb_fts)`
	pragmaStructuredMemoryTableInfoQuery = `PRAGMA table_info(structured_memory)`
)

var indexSteps []schemaStep

var contextSteps = []schemaStep{
	{version: 1, name: "normalize_timestamps", run: execSteps(normalizeContextNotesLastAccessedQuery)},
	{version: 2, name: "context_note_tombstones", run: execSteps(createContextNoteTombstonesTable)},
	{version: 3, name: "recount_token_est", run: recountTokenEstimates},
	{version: 4, name: "structured_memory_invalidation", run: execSteps(addStructuredMemoryInvalidatedReasonColumn, addStructuredMemoryWorldValidAtColumn, addStructuredMemoryWorldInvalidAtColumn, addStructuredMemoryExpiresAtColumn, createStructuredMemoryExpiresIndex)},
	{version: 5, name: "structured_memory_quarantine", run: execSteps(addStructuredMemoryQuarantinedColumn)},
	{version: 6, name: "structured_memory_kb", run: migrateStructuredMemoryKB},
}

func migrateStructuredMemoryKB(tx *sql.Tx) error {
	existing := map[string]bool{}
	rows, err := tx.Query(pragmaStructuredMemoryTableInfoQuery)
	if err != nil {
		return err
	}
	for rows.Next() {
		var cid int
		var name, colType string
		var notNull int
		var dflt sql.NullString
		var pk int
		if err := rows.Scan(&cid, &name, &colType, &notNull, &dflt, &pk); err != nil {
			rows.Close()
			return err
		}
		existing[name] = true
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()

	// Fresh databases already have these columns from initContextSchema, so skip any
	// column that already exists while adding missing ones on pre-v6 databases.
	columns := []struct {
		name string
		stmt string
	}{
		{"title", addStructuredMemoryTitleColumn},
		{"body", addStructuredMemoryBodyColumn},
		{"tags", addStructuredMemoryTagsColumn},
		{"actor", addStructuredMemoryActorColumn},
		{"category", addStructuredMemoryCategoryColumn},
		{"severity", addStructuredMemorySeverityColumn},
		{"target", addStructuredMemoryTargetColumn},
		{"before_value", addStructuredMemoryBeforeValueColumn},
		{"after_value", addStructuredMemoryAfterValueColumn},
		{"reason", addStructuredMemoryReasonColumn},
		{"occurred_at", addStructuredMemoryOccurredAtColumn},
		{"verified", addStructuredMemoryVerifiedColumn},
		{"verified_at", addStructuredMemoryVerifiedAtColumn},
		{"redacted_fields", addStructuredMemoryRedactedFieldsColumn},
	}
	for _, col := range columns {
		if existing[col.name] {
			continue
		}
		if _, err := tx.Exec(col.stmt); err != nil {
			return err
		}
	}
	return execSteps(
		createStructuredMemoryKindOccurredIndex,
		createStructuredMemoryTargetIndex,
		createStructuredMemoryKBFTSStepTable,
		backfillStructuredMemoryKBFTSQuery,
	)(tx)
}

var usageSteps = []schemaStep{
	{version: 1, name: "normalize_timestamps", run: execSteps(
		normalizeContextNoteAccessAtQuery, normalizeSessionStatsLastStoreQuery, normalizeSessionStatsLastAccessQuery,
	)},
	{version: 2, name: "queries_estimate_method", run: execSteps(addQueriesEstimateMethodColumn)},
	{version: 3, name: "queries_ledgers", run: execSteps(addQueriesConservativeBaselineColumn, addQueriesLedgerColumn)},
	{version: 4, name: "host_usage", run: execSteps(createHostUsageOffsetsTable, createHostUsageDailyTable)},
}
