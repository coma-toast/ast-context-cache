package db

import (
	"database/sql"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/tokens"
)

// TL-4: stored token_est values written before the tokenizer landed were len/4 estimates.
// The recount pages by rowid in batches of recountBatchSize inside the step's transaction.
const (
	recountBatchSize               = 500
	selectContextNotesRecountQuery = `SELECT rowid, content FROM context_notes WHERE rowid > ? ORDER BY rowid LIMIT ?`
	updateContextNoteTokensQuery   = `UPDATE context_notes SET token_est = ? WHERE rowid = ?`
	selectRevisionsRecountQuery    = `SELECT rowid, content FROM context_note_revisions WHERE rowid > ? ORDER BY rowid LIMIT ?`
	updateRevisionTokensQuery      = `UPDATE context_note_revisions SET token_est = ? WHERE rowid = ?`
	selectMemoryRecountQuery       = `SELECT rowid, kind, COALESCE(subject,''), COALESCE(predicate,''), COALESCE(object,''), COALESCE(rule,'')
		FROM structured_memory WHERE rowid > ? ORDER BY rowid LIMIT ?`
	updateMemoryTokensQuery = `UPDATE structured_memory SET token_est = ? WHERE rowid = ?`
	memoryKindProcedure     = "procedure"
	memoryKindFact          = "fact"
)

type recountRow struct {
	rowid int64
	text  string
}

// recountTokenEstimates is context step TL-4.
func recountTokenEstimates(tx *sql.Tx) error {
	if err := recountTable(tx, selectContextNotesRecountQuery, updateContextNoteTokensQuery, scanContentRow); err != nil {
		return err
	}
	if err := recountTable(tx, selectRevisionsRecountQuery, updateRevisionTokensQuery, scanContentRow); err != nil {
		return err
	}
	return recountTable(tx, selectMemoryRecountQuery, updateMemoryTokensQuery, scanMemoryRow)
}

func recountTable(tx *sql.Tx, selectQuery, updateQuery string, scan func(*sql.Rows) (recountRow, error)) error {
	var last int64
	for {
		batch, err := readRecountBatch(tx, selectQuery, last, scan)
		if err != nil || len(batch) == 0 {
			return err
		}
		for _, r := range batch {
			if _, err := tx.Exec(updateQuery, tokens.Count(r.text), r.rowid); err != nil {
				return err
			}
		}
		last = batch[len(batch)-1].rowid
	}
}

func readRecountBatch(tx *sql.Tx, selectQuery string, after int64, scan func(*sql.Rows) (recountRow, error)) ([]recountRow, error) {
	rows, err := tx.Query(selectQuery, after, recountBatchSize)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []recountRow
	for rows.Next() {
		r, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func scanContentRow(rows *sql.Rows) (recountRow, error) {
	var r recountRow
	err := rows.Scan(&r.rowid, &r.text)
	return r, err
}

func scanMemoryRow(rows *sql.Rows) (recountRow, error) {
	var r recountRow
	var kind, subj, pred, obj, rule string
	err := rows.Scan(&r.rowid, &kind, &subj, &pred, &obj, &rule)
	r.text = memoryLine(kind, subj, pred, obj, rule)
	return r, err
}

// memoryLine mirrors memory.FormatLine, which db cannot import, so the recount matches
// what the memory package counts for new rows.
func memoryLine(kind, subj, pred, obj, rule string) string {
	switch kind {
	case memoryKindProcedure:
		if rule != "" {
			return "PROC: " + strings.TrimSpace(rule)
		}
		return "PROC: (empty)"
	case memoryKindFact:
		subj, pred, obj = strings.TrimSpace(subj), strings.TrimSpace(pred), strings.TrimSpace(obj)
		if subj == "" && obj == "" {
			return ""
		}
		if pred == "" {
			pred = "is"
		}
		if subj == "" {
			return pred + " " + obj
		}
		if obj == "" {
			return subj + " " + pred
		}
		return subj + " " + pred + " " + obj
	default:
		return ""
	}
}
