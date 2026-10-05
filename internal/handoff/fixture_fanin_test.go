package handoff

import (
	"database/sql"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

// seededHandoff is one handoff seeded straight into the tables (Create and Open are built in
// parallel), with its open children.
type seededHandoff struct {
	tree     TreeID
	root     SessionID
	ref      HandoffRef
	children []SessionID
}

// handoffSeed describes a handoff to seed. With tree empty a new tree rooted at root is made;
// a nested handoff names its creating child in parentChild and its parent's tree.
type handoffSeed struct {
	root        SessionID
	tree        TreeID
	parentChild SessionID
	depth       int
	label       string
	children    int
	at          time.Time
}

// newFanInService opens a fresh database and returns a service without background loops.
func newFanInService(t *testing.T) *realService {
	t.Helper()
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)
	return newService(nil)
}

// seedHandoff inserts the tree (unless given), the handoff, and its open children in one
// handoff transaction, the way Create and Open would.
func seedHandoff(t *testing.T, in handoffSeed) seededHandoff {
	t.Helper()
	if in.at.IsZero() {
		in.at = nowFunc()
	}
	if in.depth == 0 {
		in.depth = 1
	}
	ref, err := NewHandoffRef()
	require.NoError(t, err)
	out := seededHandoff{tree: in.tree, root: in.root, ref: ref}
	if out.tree == "" {
		out.tree, err = NewTreeID()
		require.NoError(t, err)
	}
	parent := in.root
	var parentChild any
	if in.parentChild != "" {
		parent, parentChild = in.parentChild, string(in.parentChild)
	}
	ts := sqlTime(in.at)
	err = db.HandoffTx(func(tx *sql.Tx) error {
		if in.tree == "" {
			if _, err := tx.Exec(`INSERT INTO handoff_trees (tree_id, root_session_id, project_path, created_at, last_access_at)
				VALUES (?, ?, '/p', ?, ?)`, out.tree, in.root, ts, ts); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(`INSERT INTO handoffs (ref, tree_id, parent_session_id, parent_child_session_id, depth, mode, label,
			brief, project_path, child_count, created_at, last_access_at) VALUES (?, ?, ?, ?, ?, 'fresh', ?, 'brief', '/p', ?, ?, ?)`,
			ref, out.tree, parent, parentChild, in.depth, in.label, in.children, ts, ts); err != nil {
			return err
		}
		for i := range in.children {
			child := ChildSessionID(ref, i+1)
			out.children = append(out.children, child)
			if _, err := tx.Exec(`INSERT INTO handoff_children (child_session_id, handoff_ref, tree_id, status, project_path,
				opened_at, last_activity_at) VALUES (?, ?, ?, 'open', '/p', ?, ?)`, child, ref, out.tree, ts, ts); err != nil {
				return err
			}
		}
		return nil
	})
	require.NoError(t, err)
	return out
}

// queryString reads one text value from context.db.
func queryString(t *testing.T, q string, args ...any) string {
	t.Helper()
	var s sql.NullString
	require.NoError(t, db.ContextDB.QueryRow(q, args...).Scan(&s), q)
	return s.String
}

// woken reports whether ch is closed, without blocking.
func woken(ch <-chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}
