package db

import (
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	createSymbolsFTSInsertTrigger = `CREATE TRIGGER IF NOT EXISTS symbols_fts_ins AFTER INSERT ON symbols BEGIN
		INSERT INTO symbols_fts(rowid, name, fqn, code) VALUES (new.id, new.name, new.fqn, new.code);
	END`
	createSymbolsFTSDeleteTrigger = `CREATE TRIGGER IF NOT EXISTS symbols_fts_del AFTER DELETE ON symbols BEGIN
		INSERT INTO symbols_fts(symbols_fts, rowid, name, fqn, code) VALUES('delete', old.id, old.name, old.fqn, old.code);
	END`
	createSymbolsTrigramInsertTrigger = `CREATE TRIGGER IF NOT EXISTS symbols_trigram_ins AFTER INSERT ON symbols BEGIN
		INSERT INTO symbols_trigram(rowid, name, fqn) VALUES (new.id, new.name, new.fqn);
	END`
	createSymbolsTrigramDeleteTrigger = `CREATE TRIGGER IF NOT EXISTS symbols_trigram_del AFTER DELETE ON symbols BEGIN
		INSERT INTO symbols_trigram(symbols_trigram, rowid, name, fqn) VALUES('delete', old.id, old.name, old.fqn);
	END`
	rebuildFTSTableQueryTemplate     = `INSERT INTO %[1]s(%[1]s) VALUES('rebuild')`
	dropTriggerIfExistsQueryTemplate = "DROP TRIGGER IF EXISTS %s"
	selectSymbolsTriggersQuery       = `SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'symbols'`
	selectFTSDriftCountsQuery        = `SELECT
		(SELECT count(*) FROM symbols s WHERE NOT EXISTS (SELECT 1 FROM symbols_fts_docsize d WHERE d.id = s.id)),
		(SELECT count(*) FROM symbols_fts_docsize d WHERE NOT EXISTS (SELECT 1 FROM symbols s WHERE s.id = d.id)),
		(SELECT count(*) FROM symbols s WHERE NOT EXISTS (SELECT 1 FROM symbols_trigram_docsize d WHERE d.id = s.id)),
		(SELECT count(*) FROM symbols_trigram_docsize d WHERE NOT EXISTS (SELECT 1 FROM symbols s WHERE s.id = d.id))`
)

// symbolFTSTables are the full-text indexes over symbols. Both use symbols as
// external content, so they only see a row when a trigger below (or a 'rebuild')
// writes it into them.
var symbolFTSTables = []string{"symbols_fts", "symbols_trigram"}

// symbolFTSTriggers keep symbolFTSTables in step with every row inserted into or
// deleted from symbols. With any of them missing, symbol writes skip that index and
// BM25/trigram search silently degrades.
var symbolFTSTriggers = []struct{ name, ddl string }{
	{"symbols_fts_ins", createSymbolsFTSInsertTrigger},
	{"symbols_fts_del", createSymbolsFTSDeleteTrigger},
	{"symbols_trigram_ins", createSymbolsTrigramInsertTrigger},
	{"symbols_trigram_del", createSymbolsTrigramDeleteTrigger},
}

// execer is satisfied by both *sql.DB and *sql.Tx.
type execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}

func createFTSTriggers(e execer) error {
	for _, t := range symbolFTSTriggers {
		if _, err := e.Exec(t.ddl); err != nil {
			return errs.WrapMessage("failed to create trigger", err, "trigger", t.name)
		}
	}
	return nil
}

func rebuildFTSTable(e execer, table string) error {
	if _, err := e.Exec(fmt.Sprintf(rebuildFTSTableQueryTemplate, table)); err != nil {
		return errs.WrapMessage("failed to rebuild FTS table", err, "table", table)
	}
	return nil
}

// EnsureFTSTriggers recreates any missing symbol FTS trigger. It returns an error
// rather than skipping when the index is unavailable (e.g. quiesced for WAL
// maintenance), so a caller that needs the triggers back knows they aren't.
func EnsureFTSTriggers() error {
	return IndexWrite(func(tx *sql.Tx) error { return createFTSTriggers(tx) })
}

// WithoutFTSTriggers runs bulkDelete inside tx with the symbol FTS triggers
// dropped, then rebuilds both full-text indexes and recreates the triggers in the
// same transaction. One rebuild is far cheaper than firing a trigger per removed
// row when most of symbols is going away.
//
// Because DDL is transactional in SQLite, no other connection ever observes the
// triggers missing, and any error (returned here, so IndexWrite rolls tx back)
// restores them along with the deleted rows.
func WithoutFTSTriggers(tx *sql.Tx, bulkDelete func() error) error {
	for _, t := range symbolFTSTriggers {
		if _, err := tx.Exec(fmt.Sprintf(dropTriggerIfExistsQueryTemplate, t.name)); err != nil {
			return errs.WrapMessage("failed to drop trigger", err, "trigger", t.name)
		}
	}
	if err := bulkDelete(); err != nil {
		return err
	}
	for _, table := range symbolFTSTables {
		if err := rebuildFTSTable(tx, table); err != nil {
			return err
		}
	}
	return createFTSTriggers(tx)
}

// FTSHealth reports how far the symbol full-text indexes have drifted from symbols.
type FTSHealth struct {
	MissingTriggers []string
	// *Missing counts symbols rows absent from that index; *Orphans counts index
	// entries whose symbols row no longer exists.
	FTSMissing, FTSOrphans         int
	TrigramMissing, TrigramOrphans int
}

func (h FTSHealth) ftsDrift() bool     { return h.FTSMissing+h.FTSOrphans > 0 }
func (h FTSHealth) trigramDrift() bool { return h.TrigramMissing+h.TrigramOrphans > 0 }

func missingFTSTriggers(conn *sql.DB) ([]string, error) {
	rows, err := conn.Query(selectSymbolsTriggersQuery)
	if err != nil {
		return nil, errs.WrapMessage("failed to list symbols triggers", err)
	}
	defer rows.Close()
	present := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, err
		}
		present[name] = true
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	var missing []string
	for _, t := range symbolFTSTriggers {
		if !present[t.name] {
			missing = append(missing, t.name)
		}
	}
	return missing, nil
}

// MeasureFTSHealth checks the symbol FTS triggers and, if measureDrift is set,
// compares symbols ids against each index's docsize ids. The drift comparison
// reads every symbol id (about 4s on a 5.7 GB index); the trigger check is a
// sqlite_master lookup. Read-only.
func MeasureFTSHealth(measureDrift bool) (FTSHealth, error) {
	var h FTSHealth
	conn, err := IndexReader()
	if err != nil {
		return h, err
	}
	if h.MissingTriggers, err = missingFTSTriggers(conn); err != nil {
		return h, err
	}
	if !measureDrift {
		return h, nil
	}
	// One statement, so all four counts come from the same read snapshot.
	err = conn.QueryRow(selectFTSDriftCountsQuery).Scan(&h.FTSMissing, &h.FTSOrphans, &h.TrigramMissing, &h.TrigramOrphans)
	if err != nil {
		return h, errs.WrapMessage("failed to measure FTS drift", err)
	}
	return h, nil
}

// CheckFTSHealth is the self-heal pass: it recreates missing symbol FTS triggers
// and rebuilds any index that has drifted from symbols, logging what it found.
// Missing triggers always force a drift measurement, since every symbol write made
// while they were gone skipped the indexes. Returns the health found before repair.
func CheckFTSHealth(measureDrift bool) (FTSHealth, error) {
	h, err := MeasureFTSHealth(false)
	if err != nil {
		return h, err
	}
	if len(h.MissingTriggers) > 0 {
		logger.Warn("FTS self-check found missing symbols triggers, recreating them", "triggers", h.MissingTriggers)
		if err := EnsureFTSTriggers(); err != nil {
			return h, errs.WrapMessage("failed to recreate FTS triggers", err)
		}
		measureDrift = true
	}
	if !measureDrift {
		return h, nil
	}
	missing := h.MissingTriggers
	if h, err = MeasureFTSHealth(true); err != nil {
		return h, err
	}
	h.MissingTriggers = missing
	for _, idx := range []struct {
		table            string
		drift            bool
		missing, orphans int
	}{
		{"symbols_fts", h.ftsDrift(), h.FTSMissing, h.FTSOrphans},
		{"symbols_trigram", h.trigramDrift(), h.TrigramMissing, h.TrigramOrphans},
	} {
		if !idx.drift {
			continue
		}
		logger.Warn("FTS self-check found index drifted from symbols, rebuilding", "table", idx.table, "unindexed_rows", idx.missing, "orphan_entries", idx.orphans)
		start := time.Now()
		if err := IndexWrite(func(tx *sql.Tx) error { return rebuildFTSTable(tx, idx.table) }); err != nil {
			return h, err
		}
		logger.Info("FTS self-check rebuilt index", "table", idx.table, "duration", time.Since(start).Round(time.Millisecond))
	}
	return h, nil
}

const (
	// ftsTriggerCheckInterval matches the deleted-project sweep, the main thing that
	// rewrites the triggers at runtime.
	ftsTriggerCheckInterval = 5 * time.Minute
	ftsDriftCheckInterval   = time.Hour
)

var ftsSelfCheckOnce sync.Once

// StartFTSSelfCheck runs CheckFTSHealth on a ticker: the trigger check every tick,
// the drift comparison hourly. Init already recreates the triggers and rebuilds
// both indexes at startup, so the first pass waits a full tick.
func StartFTSSelfCheck() {
	ftsSelfCheckOnce.Do(func() {
		go func() {
			ticker := time.NewTicker(ftsTriggerCheckInterval)
			defer ticker.Stop()
			lastDrift := time.Now()
			for range ticker.C {
				if IndexReadQuiesced() {
					continue
				}
				due := time.Since(lastDrift) >= ftsDriftCheckInterval
				if _, err := CheckFTSHealth(due); err != nil {
					logger.Warn("FTS self-check failed", "error", err)
					continue
				}
				if due {
					lastDrift = time.Now()
				}
			}
		}()
	})
}
