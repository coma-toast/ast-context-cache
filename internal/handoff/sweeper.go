package handoff

import (
	"context"
	"database/sql"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// The first sweep waits for startup to settle; after that RQ-2 asks for at least hourly.
	sweepInitialDelay = 2 * time.Minute
	sweepInterval     = time.Hour
	abandonInterval   = time.Minute

	selectExpiredTreesQuery     = `SELECT tree_id FROM handoff_trees WHERE last_access_at < ?`
	selectInactiveChildrenQuery = `SELECT child_session_id, tree_id FROM handoff_children
		WHERE status = '` + string(StatusOpen) + `' AND last_activity_at < ?`
	markChildAbandonedQuery = `UPDATE handoff_children SET status = '` + string(StatusAbandoned) + `'
		WHERE child_session_id = ? AND status = '` + string(StatusOpen) + `'`
)

// abandonedChild is an open child whose inactivity window ran out.
type abandonedChild struct {
	sid  SessionID
	tree TreeID
}

// sweepLoop expires trees past their TTL, first shortly after start and then hourly (RQ-1, RQ-2).
func (s *realService) sweepLoop(ctx context.Context) {
	s.logger.Debug("Starting handoff sweep loop")
	defer s.logger.Debug("Stopped handoff sweep loop")
	select {
	case <-ctx.Done():
		return
	case <-time.After(sweepInitialDelay):
	}
	for {
		if _, err := s.sweep(); err != nil {
			s.logger.Warn("Failed to sweep expired handoff trees", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(sweepInterval):
		}
	}
}

// abandonLoop marks inactive children abandoned every minute (FI-3).
func (s *realService) abandonLoop(ctx context.Context) {
	s.logger.Debug("Starting handoff abandonment loop")
	defer s.logger.Debug("Stopped handoff abandonment loop")
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(abandonInterval):
		}
		if _, err := s.markAbandoned(); err != nil {
			s.logger.Warn("Failed to mark inactive handoff children abandoned", "error", err)
		}
	}
}

// sweep flushes every tree not accessed within the TTL and returns how many it flushed. One
// tree failing doesn't stop the rest; the first error is returned after all were tried.
func (s *realService) sweep() (int, error) {
	if db.ContextDB == nil {
		return 0, errNoContextDB
	}
	ttl := LoadLimits().TTL()
	rows, err := db.ContextDB.Query(selectExpiredTreesQuery, sqlTime(nowFunc().Add(-ttl)))
	if err != nil {
		return 0, errs.WrapMessage("failed to list expired handoff trees", err)
	}
	var trees []TreeID
	for rows.Next() {
		var tree TreeID
		if rows.Scan(&tree) == nil {
			trees = append(trees, tree)
		}
	}
	rows.Close()
	var firstErr error
	flushed := 0
	for _, tree := range trees {
		if _, err := s.flushTree(tree); err != nil {
			s.logger.Warn("Failed to expire handoff tree", tree.Attr(), "error", err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		flushed++
	}
	s.pruneTrail(ttl)
	if flushed > 0 {
		s.logger.Info("Expired handoff trees", "trees", flushed, "ttl", ttl)
	}
	return flushed, firstErr
}

// markAbandoned marks open children inactive past the window as abandoned, releasing their
// claims in the same transaction, and returns how many it marked. Activity (Touch) revives them.
func (s *realService) markAbandoned() (int, error) {
	cutoff := sqlTime(nowFunc().Add(-LoadLimits().ChildInactive()))
	var marked []abandonedChild
	err := db.HandoffTx(func(tx *sql.Tx) error {
		marked = nil
		children, err := inactiveChildrenTx(tx, cutoff)
		if err != nil {
			return err
		}
		for _, c := range children {
			res, err := tx.Exec(markChildAbandonedQuery, string(c.sid))
			if err != nil {
				return errs.WrapMessage("failed to mark child abandoned", err, "session", string(c.sid))
			}
			if n, _ := res.RowsAffected(); n == 0 {
				continue
			}
			if _, err := s.releaseAllTx(tx, c.tree, c.sid); err != nil {
				return err
			}
			marked = append(marked, c)
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	for _, c := range marked {
		s.waiters.notify(c.tree)
		s.logger.Info("Marked handoff child abandoned", c.sid.Attr(), c.tree.Attr())
	}
	return len(marked), nil
}

// releaseAllTx releases every claim sid holds in tree and leaves its queue positions, granting
// each key to the next waiter, and returns the released keys. Claims land in Phase 7.3; until
// then a session holds none.
func (s *realService) releaseAllTx(tx *sql.Tx, tree TreeID, sid SessionID) ([]string, error) {
	return nil, nil
}

// pruneTrail drops search-trail rows older than the tree TTL. internal/trail lands with
// Phase 5; wire trail.PruneOlderThan here when it does.
func (s *realService) pruneTrail(ttl time.Duration) {}

func inactiveChildrenTx(tx *sql.Tx, cutoff string) ([]abandonedChild, error) {
	rows, err := tx.Query(selectInactiveChildrenQuery, cutoff)
	if err != nil {
		return nil, errs.WrapMessage("failed to list inactive handoff children", err)
	}
	defer rows.Close()
	var out []abandonedChild
	for rows.Next() {
		var c abandonedChild
		if err := rows.Scan(&c.sid, &c.tree); err != nil {
			return nil, errs.WrapMessage("failed to read inactive handoff child", err)
		}
		out = append(out, c)
	}
	return out, rows.Err()
}
