package db

import (
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/tokens"
)

const (
	testInsertRecountNoteQuery     = `INSERT INTO context_notes (ref, session_id, content, content_hash, token_est) VALUES (?, 's', ?, 'h', 1)`
	testInsertRecountRevisionQuery = `INSERT INTO context_note_revisions (ref, revision, content, token_est) VALUES (?, 1, ?, 1)`
	testInsertRecountMemoryQuery   = `INSERT INTO structured_memory (ref, kind, subject, predicate, object, rule, token_est) VALUES (?, ?, ?, ?, ?, ?, 1)`
	testSelectNoteTokensQuery      = `SELECT token_est FROM context_notes WHERE ref = ?`
	testSelectRevisionTokensQuery  = `SELECT token_est FROM context_note_revisions WHERE ref = ?`
	testSelectMemoryTokensQuery    = `SELECT token_est FROM structured_memory WHERE ref = ?`
	testSelectQueryLedgerQuery     = `SELECT ledger, conservative_baseline_tokens FROM queries WHERE tool_name = ?`
)

func TestRecountTokenEstimates(t *testing.T) {
	conn, err := openStepPool(filepath.Join(t.TempDir(), "ctx.db"))
	require.NoError(t, err)
	defer conn.Close()
	for _, q := range []string{createContextNotesTable, createContextNoteRevisionsTable, createStructuredMemoryTable} {
		_, err := conn.Exec(q)
		require.NoError(t, err)
	}
	// More rows than one batch, so paging is exercised.
	for i := range recountBatchSize + 3 {
		_, err := conn.Exec(testInsertRecountNoteQuery, "ctx_"+strconv.Itoa(i), "func main() { fmt.Println(\"hello\") }")
		require.NoError(t, err)
	}
	_, err = conn.Exec(testInsertRecountRevisionQuery, "ctx_0", "an older revision body")
	require.NoError(t, err)
	_, err = conn.Exec(testInsertRecountMemoryQuery, "mem_f", memoryKindFact, "repo", "", "uses testify", "")
	require.NoError(t, err)
	_, err = conn.Exec(testInsertRecountMemoryQuery, "mem_p", memoryKindProcedure, "", "", "", "  run make test  ")
	require.NoError(t, err)
	tx, err := conn.Begin()
	require.NoError(t, err)
	require.NoError(t, recountTokenEstimates(tx))
	require.NoError(t, tx.Commit())
	var n int
	for _, ref := range []string{"ctx_0", "ctx_" + strconv.Itoa(recountBatchSize+2)} {
		require.NoError(t, conn.QueryRow(testSelectNoteTokensQuery, ref).Scan(&n))
		assert.Equal(t, tokens.Count("func main() { fmt.Println(\"hello\") }"), n, ref)
	}
	require.NoError(t, conn.QueryRow(testSelectRevisionTokensQuery, "ctx_0").Scan(&n))
	assert.Equal(t, tokens.Count("an older revision body"), n)
	require.NoError(t, conn.QueryRow(testSelectMemoryTokensQuery, "mem_f").Scan(&n))
	assert.Equal(t, tokens.Count("repo is uses testify"), n)
	require.NoError(t, conn.QueryRow(testSelectMemoryTokensQuery, "mem_p").Scan(&n))
	assert.Equal(t, tokens.Count("PROC: run make test"), n)
}

func TestQueryLogWritesLedgerColumns(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	t.Cleanup(Close)
	require.NoError(t, Init())
	LogQuery("ledger_tool", nil, QueryLogMetrics{ConservativeBaseline: 321, Ledger: "compression"}, "/p", "")
	FlushWriteBuffers()
	var ledger string
	var conservative int
	require.NoError(t, DB.QueryRow(testSelectQueryLedgerQuery, "ledger_tool").Scan(&ledger, &conservative))
	assert.Equal(t, "compression", ledger)
	assert.Equal(t, 321, conservative)
}
