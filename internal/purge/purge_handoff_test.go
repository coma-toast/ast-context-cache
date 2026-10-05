package purge

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func seedHandoffTree(t *testing.T, tree, project string) {
	t.Helper()
	ref, child := "hof_"+tree, "hof_"+tree+".c1"
	for _, q := range []struct {
		sql  string
		args []any
	}{
		{`INSERT INTO handoff_trees (tree_id, root_session_id, project_path) VALUES (?, 'root', ?)`, []any{tree, project}},
		{`INSERT INTO handoffs (ref, tree_id, parent_session_id, brief, project_path) VALUES (?, ?, 'root', 'b', ?)`, []any{ref, tree, project}},
		{`INSERT INTO handoff_snapshot_items (handoff_ref, section) VALUES (?, 'trail')`, []any{ref}},
		{`INSERT INTO handoff_children (child_session_id, handoff_ref, tree_id, project_path) VALUES (?, ?, ?, ?)`, []any{child, ref, tree, project}},
		{`INSERT INTO handoff_results (child_session_id, result_ref) VALUES (?, 'ctx_r')`, []any{child}},
		{`INSERT INTO scratchpad_entries (tree_id, author_session_id, type, text) VALUES (?, ?, 'finding', 'x')`, []any{tree, child}},
		{`INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES (?, 'a.go', ?)`, []any{tree, child}},
		{`INSERT INTO handoff_claim_queue (tree_id, key, session_id) VALUES (?, 'a.go', 'other')`, []any{tree}},
		{`INSERT INTO handoff_claim_grants (session_id, tree_id, key) VALUES (?, ?, 'b.go')`, []any{child, tree}},
	} {
		_, err := db.ContextDB.Exec(q.sql, q.args...)
		require.NoError(t, err, q.sql)
	}
}

func handoffRowCounts(t *testing.T) map[string]int {
	t.Helper()
	out := map[string]int{}
	for _, table := range []string{
		"handoff_trees", "handoffs", "handoff_snapshot_items", "handoff_children", "handoff_results",
		"scratchpad_entries", "handoff_claims", "handoff_claim_queue", "handoff_claim_grants",
	} {
		var n int
		require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM `+table).Scan(&n))
		out[table] = n
	}
	return out
}

func TestProjectDataDeletesHandoffTrees(t *testing.T) {
	home := dbtest.Init(t)
	gone, kept := filepath.Join(home, "gone"), filepath.Join(home, "kept")
	seedHandoffTree(t, "hft_gone", gone)
	seedHandoffTree(t, "hft_kept", kept)
	require.NoError(t, ProjectData(gone))
	for table, n := range handoffRowCounts(t) {
		assert.Equal(t, 1, n, "%s keeps only the other project's tree", table)
	}
	var tree string
	require.NoError(t, db.ContextDB.QueryRow(`SELECT tree_id FROM handoff_trees`).Scan(&tree))
	assert.Equal(t, "hft_kept", tree)
}
