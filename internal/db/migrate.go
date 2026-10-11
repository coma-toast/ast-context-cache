package db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	selectUserVersionQuery      = `PRAGMA user_version`
	setUserVersionQueryTemplate = `PRAGMA user_version = %d`
	foreignKeyCheckQuery        = `PRAGMA foreign_key_check`
	countSchemaObjectsQuery     = `SELECT COUNT(*) FROM sqlite_master`
	snapshotDSNParams           = "?_busy_timeout=15000"
	snapshotsDirName            = "snapshots"
	pre50SnapshotDirName        = "pre-5.0"
)

// schemaStep is one versioned, additive schema change. The init*Schema functions stay the
// idempotent baseline; steps run after them, in order, each in its own transaction that
// also sets PRAGMA user_version to the step's version.
type schemaStep struct {
	version int
	name    string
	run     func(tx *sql.Tx) error
}

var fkViolations atomic.Int64

// ForeignKeyViolations reports how many rows PRAGMA foreign_key_check found in context.db
// on the first start that enforced foreign keys. They are reported, never deleted.
func ForeignKeyViolations() int {
	return int(fkViolations.Load())
}

// runSteps applies each step whose version is above the database's user_version. conn
// should begin transactions IMMEDIATE (openStepPool), so a step never has to upgrade a
// read lock; the version is re-read inside the transaction in case another process
// applied the step first.
func runSteps(conn *sql.DB, dbName string, steps []schemaStep) error {
	current, err := userVersion(conn)
	if err != nil {
		return errs.WrapMessage("failed to read schema version", err, "db", dbName)
	}
	for _, s := range steps {
		if s.version <= current {
			continue
		}
		if err := runStep(conn, s); err != nil {
			return errs.WrapMessage("schema step failed", err, "db", dbName, "step", s.name)
		}
		logger.Info("Applied schema step", "db", dbName, "step", s.name, "version", s.version)
	}
	return nil
}

func runStep(conn *sql.DB, s schemaStep) error {
	tx, err := conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var v int
	if err := tx.QueryRow(selectUserVersionQuery).Scan(&v); err != nil {
		return err
	}
	if v >= s.version {
		return nil
	}
	if err := s.run(tx); err != nil {
		return err
	}
	if _, err := tx.Exec(fmt.Sprintf(setUserVersionQueryTemplate, s.version)); err != nil {
		return err
	}
	return tx.Commit()
}

func userVersion(conn *sql.DB) (int, error) {
	var v int
	err := conn.QueryRow(selectUserVersionQuery).Scan(&v)
	return v, err
}

// execSteps returns a step body that runs queries in order.
func execSteps(queries ...string) func(tx *sql.Tx) error {
	return func(tx *sql.Tx) error {
		for _, q := range queries {
			if _, err := tx.Exec(q); err != nil {
				return err
			}
		}
		return nil
	}
}

// migrateDB runs steps against the database at path through its own IMMEDIATE pool.
func migrateDB(path, dbName string, steps []schemaStep) error {
	if len(steps) == 0 {
		return nil
	}
	conn, err := openStepPool(path)
	if err != nil {
		return fmtOpenErr(dbName, path, err)
	}
	defer conn.Close()
	return runSteps(conn, dbName, steps)
}

// migrateSchemas runs every database's pending steps. Before context.db's first step it
// reports existing foreign-key violations, which enforcement (connPragmas) now surfaces.
func migrateSchemas(idxPath, ctxPath, usePath string) error {
	if err := migrateDB(idxPath, "index", indexSteps); err != nil {
		return err
	}
	if v, err := userVersion(ContextDB); err == nil && v < 1 {
		checkForeignKeys(ContextDB)
	}
	if err := migrateDB(ctxPath, "context", contextSteps); err != nil {
		return err
	}
	return migrateDB(usePath, "usage", usageSteps)
}

func checkForeignKeys(conn *sql.DB) {
	rows, err := conn.Query(foreignKeyCheckQuery)
	if err != nil {
		logger.Warn("Failed to run foreign key check", "db", "context", "error", err)
		return
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		n++
	}
	if err := rows.Err(); err != nil {
		logger.Warn("Failed to read foreign key check", "db", "context", "error", err)
	}
	fkViolations.Store(int64(n))
	if n > 0 {
		logger.Warn("Foreign key violations found, left in place", "db", "context", "violations", n)
	}
}

func pre50SnapshotDir() string {
	return filepath.Join(cacheDir(), snapshotsDirName, pre50SnapshotDirName)
}

// snapshotPre50 copies a 4.x database (non-empty, user_version 0) to
// <dataDir>/snapshots/pre-5.0/<dbName>.db before any schema step touches it. It runs once
// per database: an existing snapshot is kept, and a database already at version 1 or
// later, or with no tables yet (needsSplitMigration's probe can leave a bare WAL-mode
// file behind), is skipped. The copy goes to a temp name first, so a failed copy is never
// mistaken for a finished one.
func snapshotPre50(path, dbName string) error {
	if fi, err := os.Stat(path); err != nil || fi.Size() == 0 {
		return nil
	}
	dest := filepath.Join(pre50SnapshotDir(), dbName+".db")
	if _, err := os.Stat(dest); err == nil {
		return nil
	}
	conn, err := sql.Open("sqlite3", path+snapshotDSNParams)
	if err != nil {
		return errs.WrapMessage("failed to open database for pre-5.0 snapshot", err, "db", dbName, "path", path)
	}
	defer conn.Close()
	v, err := userVersion(conn)
	if err != nil {
		return errs.WrapMessage("failed to read schema version for pre-5.0 snapshot", err, "db", dbName, "path", path)
	}
	var objects int
	if err := conn.QueryRow(countSchemaObjectsQuery).Scan(&objects); err != nil {
		return errs.WrapMessage("failed to read schema for pre-5.0 snapshot", err, "db", dbName, "path", path)
	}
	if v > 0 || objects == 0 {
		return nil
	}
	if err := os.MkdirAll(pre50SnapshotDir(), 0o755); err != nil {
		return errs.WrapMessage("failed to create pre-5.0 snapshot directory", err, "db", dbName, "dir", pre50SnapshotDir())
	}
	tmp := dest + ".tmp"
	os.Remove(tmp)
	if _, err := conn.Exec(vacuumIntoQuery, tmp); err != nil {
		os.Remove(tmp)
		return errs.WrapMessage("failed to write pre-5.0 snapshot", err, "db", dbName, "dest", dest)
	}
	if err := os.Rename(tmp, dest); err != nil {
		return errs.WrapMessage("failed to finish pre-5.0 snapshot", err, "db", dbName, "dest", dest)
	}
	logger.Info("Saved pre-5.0 database snapshot", "db", dbName, "dest", dest)
	return nil
}
