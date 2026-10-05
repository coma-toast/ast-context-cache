package db

import (
	"database/sql"
	"fmt"
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/startup"
)

const (
	countSymbolsTableQuery              = `SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='symbols'`
	countSymbolsQuery                   = `SELECT COUNT(*) FROM symbols`
	rebuildSymbolsFTSQuery              = `INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`
	rebuildSymbolsTrigramQuery          = `INSERT INTO symbols_trigram(symbols_trigram) VALUES('rebuild')`
	rebuildDocsFTSQuery                 = `INSERT INTO docs_fts(docs_fts) VALUES('rebuild')`
	rebuildContextNotesFTSQuery         = `INSERT INTO context_notes_fts(context_notes_fts) VALUES('rebuild')`
	rebuildStructuredMemoryFTSQuery     = `INSERT INTO structured_memory_fts(structured_memory_fts) VALUES('rebuild')`
	attachSrcDatabaseQueryTemplate      = `ATTACH DATABASE '%s' AS src`
	detachSrcDatabaseQuery              = `DETACH DATABASE src`
	countSrcTableQuery                  = `SELECT COUNT(*) FROM src.sqlite_master WHERE type='table' AND name=?`
	copySrcTableQueryTemplate           = `INSERT INTO main.%[1]s (%[2]s) SELECT %[2]s FROM src.%[1]s`
	selectTableColumnNamesQuery         = `SELECT name FROM pragma_table_info(?, ?)`
	dropTableIfExistsQueryTemplate      = `DROP TABLE IF EXISTS %s`
	selectSessionsMissingSymbolQuery    = `SELECT id, symbol_id FROM sessions WHERE symbol_id > 0 AND (symbol_name IS NULL OR symbol_name = '')`
	selectSymbolForSessionBackfillQuery = `SELECT name, file, COALESCE(start_line,0) FROM symbols WHERE id=?`
	updateSessionSymbolFieldsQuery      = `UPDATE sessions SET symbol_name=?, start_line=?, file_path=COALESCE(NULLIF(file_path,''), ?) WHERE id=?`
)

var indexTables = []string{
	"symbols", "edges", "summaries", "vectors", "indexed_files", "embed_pending",
}

var contextTables = []string{
	"doc_sources", "doc_content", "context_notes", "structured_memory", "kv_repair_events",
	"handoff_trees", "handoffs", "handoff_snapshot_items", "handoff_children", "handoff_results",
	"scratchpad_entries", "handoff_claims", "handoff_claim_queue", "handoff_claim_grants",
}

var (
	monolithicDropTables  = append(append([]string{}, indexTables...), contextTables...)
	monolithicDropVirtual = []string{
		"symbols_fts", "docs_fts", "context_notes_fts", "structured_memory_fts",
	}
)

func needsSplitMigration(usagePath, indexPath string) bool {
	usageConn, err := sql.Open("sqlite3", usagePath+"?mode=ro&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return false
	}
	defer usageConn.Close()
	var usageTables int
	if err := usageConn.QueryRow(countSymbolsTableQuery).Scan(&usageTables); err != nil || usageTables == 0 {
		return false
	}
	var usageSymbols int64
	if err := usageConn.QueryRow(countSymbolsQuery).Scan(&usageSymbols); err != nil || usageSymbols == 0 {
		return false
	}
	if !indexHasSymbols(indexPath) {
		return true
	}
	indexConn, err := sql.Open("sqlite3", indexPath+"?mode=ro&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return true
	}
	defer indexConn.Close()
	var indexSymbols int64
	if err := indexConn.QueryRow(countSymbolsQuery).Scan(&indexSymbols); err != nil {
		return true
	}
	if indexSymbols >= usageSymbols {
		return false
	}
	logger.Warn("Resuming incomplete split migration", "usage_symbols", usageSymbols, "index_symbols", indexSymbols)
	removePartialDB(indexPath)
	removePartialDB(contextDBPath())
	return true
}

func removePartialDB(path string) {
	os.Remove(path)
	os.Remove(path + "-wal")
	os.Remove(path + "-shm")
}

func indexHasSymbols(indexPath string) bool {
	if _, err := os.Stat(indexPath); err != nil {
		return false
	}
	conn, err := sql.Open("sqlite3", indexPath+"?mode=ro&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return false
	}
	defer conn.Close()
	var n int
	err = conn.QueryRow(countSymbolsTableQuery).Scan(&n)
	return err == nil && n > 0
}

func migrateSplitDB(usagePath, indexPath, contextPath string) error {
	logger.Info("Migrating monolithic database to index.db and context.db", "path", usagePath)
	startup.SetMessage("Migrating database (index tables)…")

	idx, err := openPool(indexPath)
	if err != nil {
		return fmtOpenErr("index", indexPath, err)
	}
	defer idx.Close()
	initIndexSchema(idx)

	ctxDB, err := openPool(contextPath)
	if err != nil {
		return fmtOpenErr("context", contextPath, err)
	}
	defer ctxDB.Close()
	initContextSchema(ctxDB)

	if err := copyTablesFromAttach(idx, usagePath, indexTables); err != nil {
		return errs.WrapMessage("failed to copy index tables in split migration", err)
	}
	idx.Exec(rebuildSymbolsFTSQuery)
	idx.Exec(rebuildSymbolsTrigramQuery)

	startup.SetMessage("Migrating database (context tables)…")
	if err := copyTablesFromAttach(ctxDB, usagePath, contextTables); err != nil {
		return errs.WrapMessage("failed to copy context tables in split migration", err)
	}
	startup.SetMessage("Migrating database (finalizing)…")
	ctxDB.Exec(rebuildDocsFTSQuery)
	ctxDB.Exec(rebuildContextNotesFTSQuery)
	ctxDB.Exec(rebuildStructuredMemoryFTSQuery)

	usage, err := openPool(usagePath)
	if err != nil {
		return fmtOpenErr("usage", usagePath, err)
	}
	defer usage.Close()
	initUsageSchema(usage)
	if err := trimMonolithicTables(usage); err != nil {
		return errs.WrapMessage("failed to trim usage tables in split migration", err)
	}
	backfillSessionDedupFields(usage, idx)

	logger.Info("Split migration complete", "index_path", indexPath, "context_path", contextPath)
	return nil
}

func copyTablesFromAttach(dest *sql.DB, srcPath string, tables []string) error {
	esc := strings.ReplaceAll(srcPath, "'", "''")
	if _, err := dest.Exec(fmt.Sprintf(attachSrcDatabaseQueryTemplate, esc)); err != nil {
		return err
	}
	defer dest.Exec(detachSrcDatabaseQuery)
	for _, t := range tables {
		var n int
		if err := dest.QueryRow(countSrcTableQuery, t).Scan(&n); err != nil || n == 0 {
			continue
		}
		// Copy by name: the destination may have columns added since the
		// monolithic DB was written (e.g. indexed_files.parser_version).
		cols := strings.Join(tableColumns(dest, "src", t), ", ")
		if _, err := dest.Exec(fmt.Sprintf(copySrcTableQueryTemplate, t, cols)); err != nil {
			return errs.WrapMessage("failed to copy table", err, "table", t)
		}
		logger.Info("Copied table into split database", "table", t)
	}
	return nil
}

// tableColumns returns the columns of schema.table (quoted) that main.table also has.
func tableColumns(conn *sql.DB, schema, table string) []string {
	names := func(s string) []string {
		rows, err := conn.Query(selectTableColumnNamesQuery, table, s)
		if err != nil {
			return nil
		}
		defer rows.Close()
		var out []string
		for rows.Next() {
			var n string
			if rows.Scan(&n) == nil {
				out = append(out, n)
			}
		}
		return out
	}
	have := map[string]bool{}
	for _, n := range names("main") {
		have[n] = true
	}
	var cols []string
	for _, n := range names(schema) {
		if have[n] {
			cols = append(cols, `"`+strings.ReplaceAll(n, `"`, `""`)+`"`)
		}
	}
	return cols
}

func trimMonolithicTables(usage *sql.DB) error {
	for _, t := range monolithicDropVirtual {
		usage.Exec(fmt.Sprintf(dropTableIfExistsQueryTemplate, t))
	}
	for _, t := range monolithicDropTables {
		usage.Exec(fmt.Sprintf(dropTableIfExistsQueryTemplate, t))
	}
	logger.Info("Trimmed index and context tables from usage.db, VACUUM deferred")
	return nil
}

func backfillSessionDedupFields(usage, index *sql.DB) {
	rows, err := usage.Query(selectSessionsMissingSymbolQuery)
	if err != nil {
		return
	}
	defer rows.Close()
	type row struct {
		id, symID int
	}
	var pending []row
	for rows.Next() {
		var r row
		if rows.Scan(&r.id, &r.symID) == nil {
			pending = append(pending, r)
		}
	}
	for _, r := range pending {
		var name, file string
		var startLine int
		if index.QueryRow(selectSymbolForSessionBackfillQuery, r.symID).Scan(&name, &file, &startLine) != nil {
			continue
		}
		usage.Exec(updateSessionSymbolFieldsQuery, name, startLine, file, r.id)
	}
}
