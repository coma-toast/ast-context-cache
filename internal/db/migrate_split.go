package db

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/startup"
)

var indexTables = []string{
	"symbols", "edges", "summaries", "vectors", "indexed_files", "embed_pending",
}

var contextTables = []string{
	"doc_sources", "doc_content", "context_notes", "structured_memory", "kv_repair_events",
}

var monolithicDropTables = append(append([]string{}, indexTables...), contextTables...)
var monolithicDropVirtual = []string{
	"symbols_fts", "docs_fts", "context_notes_fts", "structured_memory_fts",
}

func needsSplitMigration(usagePath, indexPath string) bool {
	usageConn, err := sql.Open("sqlite3", usagePath+"?mode=ro&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return false
	}
	defer usageConn.Close()
	var usageTables int
	if err := usageConn.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='symbols'`).Scan(&usageTables); err != nil || usageTables == 0 {
		return false
	}
	var usageSymbols int64
	if err := usageConn.QueryRow(`SELECT COUNT(*) FROM symbols`).Scan(&usageSymbols); err != nil || usageSymbols == 0 {
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
	if err := indexConn.QueryRow(`SELECT COUNT(*) FROM symbols`).Scan(&indexSymbols); err != nil {
		return true
	}
	if indexSymbols >= usageSymbols {
		return false
	}
	log.Printf("db split: incomplete migration (usage symbols=%d index symbols=%d) — resuming", usageSymbols, indexSymbols)
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
	err = conn.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='symbols'`).Scan(&n)
	return err == nil && n > 0
}

func migrateSplitDB(usagePath, indexPath, contextPath string) error {
	log.Printf("db split: migrating monolithic %s -> index.db + context.db", usagePath)
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
		return fmt.Errorf("split migration index: %w", err)
	}
	idx.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)

	startup.SetMessage("Migrating database (context tables)…")
	if err := copyTablesFromAttach(ctxDB, usagePath, contextTables); err != nil {
		return fmt.Errorf("split migration context: %w", err)
	}
	startup.SetMessage("Migrating database (finalizing)…")
	ctxDB.Exec(`INSERT INTO docs_fts(docs_fts) VALUES('rebuild')`)
	ctxDB.Exec(`INSERT INTO context_notes_fts(context_notes_fts) VALUES('rebuild')`)
	ctxDB.Exec(`INSERT INTO structured_memory_fts(structured_memory_fts) VALUES('rebuild')`)

	usage, err := openPool(usagePath)
	if err != nil {
		return fmtOpenErr("usage", usagePath, err)
	}
	defer usage.Close()
	initUsageSchema(usage)
	if err := trimMonolithicTables(usage); err != nil {
		return fmt.Errorf("split migration trim usage: %w", err)
	}
	backfillSessionDedupFields(usage, idx)

	log.Printf("db split: migration complete (index=%s context=%s)", indexPath, contextPath)
	return nil
}

func copyTablesFromAttach(dest *sql.DB, srcPath string, tables []string) error {
	esc := strings.ReplaceAll(srcPath, "'", "''")
	if _, err := dest.Exec(`ATTACH DATABASE '` + esc + `' AS src`); err != nil {
		return err
	}
	defer dest.Exec(`DETACH DATABASE src`)
	for _, t := range tables {
		var n int
		if err := dest.QueryRow(`SELECT COUNT(*) FROM src.sqlite_master WHERE type='table' AND name=?`, t).Scan(&n); err != nil || n == 0 {
			continue
		}
		if _, err := dest.Exec(`INSERT INTO main.` + t + ` SELECT * FROM src.` + t); err != nil {
			return fmt.Errorf("copy %s: %w", t, err)
		}
		log.Printf("db split: copied table %s", t)
	}
	return nil
}

func trimMonolithicTables(usage *sql.DB) error {
	for _, t := range monolithicDropVirtual {
		usage.Exec(`DROP TABLE IF EXISTS ` + t)
	}
	for _, t := range monolithicDropTables {
		usage.Exec(`DROP TABLE IF EXISTS ` + t)
	}
	log.Printf("db split: trimmed index/context tables from usage.db (VACUUM deferred)")
	return nil
}

func backfillSessionDedupFields(usage, index *sql.DB) {
	rows, err := usage.Query(`SELECT id, symbol_id FROM sessions WHERE symbol_id > 0 AND (symbol_name IS NULL OR symbol_name = '')`)
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
		if index.QueryRow(`SELECT name, file, COALESCE(start_line,0) FROM symbols WHERE id=?`, r.symID).Scan(&name, &file, &startLine) != nil {
			continue
		}
		usage.Exec(`UPDATE sessions SET symbol_name=?, start_line=?, file_path=COALESCE(NULLIF(file_path,''), ?) WHERE id=?`,
			name, startLine, file, r.id)
	}
}
