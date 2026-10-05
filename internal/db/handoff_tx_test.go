package db_test

import (
	"database/sql"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// No t.Parallel: the db pools are package globals.

func TestHandoffSchemaTables(t *testing.T) {
	dbtest.Init(t)
	tables := []string{
		"handoff_trees", "handoffs", "handoff_snapshot_items", "handoff_children", "handoff_results",
		"scratchpad_entries", "handoff_claims", "handoff_claim_queue", "handoff_claim_grants",
	}
	for _, table := range tables {
		var n int
		require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?`, table).Scan(&n))
		assert.Equal(t, 1, n, table)
	}
	indexes := []string{
		"idx_handoffs_parent", "idx_handoffs_tree", "idx_handoff_snapshot_items_section",
		"idx_handoff_children_tree_status", "idx_scratchpad_entries_tree", "idx_handoff_claim_queue_key",
	}
	for _, idx := range indexes {
		var n int
		require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?`, idx).Scan(&n))
		assert.Equal(t, 1, n, idx)
	}
	_, err := db.ContextDB.Exec(`INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES ('t', 'k', 'a')`)
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES ('t', 'k', 'b')`)
	assert.Error(t, err, "one holder per (tree_id, key)")
}

func TestHandoffTxSerializesReadThenWrite(t *testing.T) {
	dbtest.Init(t)
	_, err := db.ContextDB.Exec(`CREATE TABLE handoff_tx_counter (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)`)
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`INSERT INTO handoff_tx_counter (id, n) VALUES (1, 0)`)
	require.NoError(t, err)
	increment := func(tx *sql.Tx) error {
		var n int
		if err := tx.QueryRow(`SELECT n FROM handoff_tx_counter WHERE id = 1`).Scan(&n); err != nil {
			return err
		}
		_, err := tx.Exec(`UPDATE handoff_tx_counter SET n = ? WHERE id = 1`, n+1)
		return err
	}
	var wg sync.WaitGroup
	errCh := make(chan error, 400)
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 200 {
				if err := db.HandoffTx(increment); err != nil {
					errCh <- err
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Errorf("HandoffTx: %v", err)
	}
	var n int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT n FROM handoff_tx_counter WHERE id = 1`).Scan(&n))
	assert.Equal(t, 400, n)
}

func TestHandoffTxRollsBackOnError(t *testing.T) {
	dbtest.Init(t)
	errBoom := errs.NewCode(errs.CodeConflict, "boom")
	err := db.HandoffTx(func(tx *sql.Tx) error {
		if _, err := tx.Exec(`INSERT INTO handoff_trees (tree_id, root_session_id) VALUES ('hft_x', 'root')`); err != nil {
			return err
		}
		return errBoom
	})
	require.ErrorIs(t, err, errBoom)
	var n int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM handoff_trees`).Scan(&n))
	assert.Zero(t, n)
	assert.Panics(t, func() {
		db.HandoffTx(func(tx *sql.Tx) error {
			tx.Exec(`INSERT INTO handoff_trees (tree_id, root_session_id) VALUES ('hft_y', 'root')`)
			panic("boom")
		})
	})
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM handoff_trees`).Scan(&n))
	assert.Zero(t, n, "a panic rolls back")
	require.NoError(t, db.HandoffTx(func(tx *sql.Tx) error { return nil }), "the connection is usable after a panic")
}

func TestSettingInt(t *testing.T) {
	dbtest.Init(t)
	const key, env = "test_setting_int", "AST_TEST_SETTING_INT"
	tests := []struct {
		name, env, setting string
		want               int
	}{
		{name: "default", want: 7},
		{name: "setting", setting: "12", want: 12},
		{name: "env beats setting", env: "3", setting: "12", want: 3},
		{name: "bad env uses default", env: "abc", setting: "12", want: 7},
		{name: "zero setting uses default", setting: "0", want: 7},
		{name: "negative env uses default", env: "-4", want: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(env, tt.env)
			require.NoError(t, db.SetSetting(key, tt.setting))
			assert.Equal(t, tt.want, db.SettingInt(key, env, 7))
		})
	}
}
