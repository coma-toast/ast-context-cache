package handoff

import (
	"database/sql"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

const (
	selectTreeChildIDsQuery = `SELECT child_session_id FROM handoff_children WHERE tree_id = ?`
	// Dependents first: the snapshot and result deletes find their rows through handoffs and
	// handoff_children, so those go after them, and the tree row last.
	deleteTreeSnapshotItemsQuery = `DELETE FROM handoff_snapshot_items WHERE handoff_ref IN (SELECT ref FROM handoffs WHERE tree_id = ?)`
	deleteTreeResultsQuery       = `DELETE FROM handoff_results WHERE child_session_id IN (SELECT child_session_id FROM handoff_children WHERE tree_id = ?)`
	deleteTreeScratchpadQuery    = `DELETE FROM scratchpad_entries WHERE tree_id = ?`
	deleteTreeClaimsQuery        = `DELETE FROM handoff_claims WHERE tree_id = ?`
	deleteTreeClaimQueueQuery    = `DELETE FROM handoff_claim_queue WHERE tree_id = ?`
	deleteTreeClaimGrantsQuery   = `DELETE FROM handoff_claim_grants WHERE tree_id = ?`
	deleteTreeChildrenQuery      = `DELETE FROM handoff_children WHERE tree_id = ?`
	deleteTreeHandoffsQuery      = `DELETE FROM handoffs WHERE tree_id = ?`
	deleteTreeQuery              = `DELETE FROM handoff_trees WHERE tree_id = ?`
)

// treeDeleteQueries run in order inside flushTree's transaction; each takes the tree id.
var treeDeleteQueries = []string{
	deleteTreeSnapshotItemsQuery, deleteTreeResultsQuery, deleteTreeScratchpadQuery,
	deleteTreeClaimsQuery, deleteTreeClaimQueueQuery, deleteTreeClaimGrantsQuery,
	deleteTreeChildrenQuery, deleteTreeHandoffsQuery, deleteTreeQuery,
}

// flushTree deletes tree and everything it owns (RQ-1, RQ-3, DC-4). The tree's rows go in one
// handoff transaction, so the tree is either intact or gone. Its child sessions' notes, memory,
// dedup rows, and search trail live outside that transaction (usage.db, index.db vectors) and
// are deleted afterwards, best effort: a failure there is logged, not returned, because the
// tree they belonged to no longer exists to retry from. Memory promoted to the parent session
// is the parent's and stays.
func (s *realService) flushTree(tree TreeID) (*FlushResponse, error) {
	res := &FlushResponse{TreeID: tree}
	var children []SessionID
	err := db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		if children, err = treeChildIDsTx(tx, tree); err != nil {
			return err
		}
		for _, q := range treeDeleteQueries {
			r, err := tx.Exec(q, string(tree))
			if err != nil {
				return errs.WrapMessage("failed to delete handoff tree rows", err, "tree", string(tree))
			}
			if q == deleteTreeHandoffsQuery {
				n, _ := r.RowsAffected()
				res.Handoffs = int(n)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	res.Children = len(children)
	sids := make([]string, len(children))
	for i, c := range children {
		sids[i] = string(c)
		if n, err := contextnotes.FlushSession(sids[i]); err != nil {
			s.logger.Warn("Failed to delete expired child session notes", c.Attr(), tree.Attr(), "error", err)
		} else {
			res.NotesDeleted += n
		}
		if n, err := memory.DeleteSession(sids[i]); err != nil {
			s.logger.Warn("Failed to delete expired child session memory", c.Attr(), tree.Attr(), "error", err)
		} else {
			res.MemoryDeleted += n
		}
	}
	if err := astcontext.DeleteSessionKeys(sids...); err != nil {
		s.logger.Warn("Failed to delete expired child dedup rows", tree.Attr(), "error", err)
	}
	s.deleteTrailSessions(children)
	s.trees.forgetTree(tree)
	s.waiters.notify(tree)
	s.logger.Info("Flushed handoff tree", tree.Attr(), "handoffs", res.Handoffs, "children", res.Children,
		"notes", res.NotesDeleted, "memory", res.MemoryDeleted)
	return res, nil
}

// deleteTrailSessions drops the search-trail rows of flushed child sessions, best effort like
// the other out-of-transaction deletes in flushTree; PruneOlderThan catches any left behind.
func (s *realService) deleteTrailSessions(sids []SessionID) {
	ids := make([]string, len(sids))
	for i, sid := range sids {
		ids[i] = string(sid)
	}
	if _, err := trail.DeleteSessions(ids...); err != nil {
		s.logger.Warn("Failed to delete flushed child search trail", "children", len(sids), "error", err)
	}
}

func treeChildIDsTx(tx *sql.Tx, tree TreeID) ([]SessionID, error) {
	rows, err := tx.Query(selectTreeChildIDsQuery, string(tree))
	if err != nil {
		return nil, errs.WrapMessage("failed to list handoff tree children", err, "tree", string(tree))
	}
	defer rows.Close()
	var out []SessionID
	for rows.Next() {
		var sid SessionID
		if err := rows.Scan(&sid); err != nil {
			return nil, errs.WrapMessage("failed to read handoff tree child", err, "tree", string(tree))
		}
		out = append(out, sid)
	}
	return out, rows.Err()
}
