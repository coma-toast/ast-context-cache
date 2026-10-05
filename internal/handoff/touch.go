package handoff

import (
	"database/sql"
	"errors"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	selectChildStatusQuery = `SELECT status FROM handoff_children WHERE child_session_id = ?`
	touchChildQuery        = `UPDATE handoff_children SET last_activity_at = ?,
		status = CASE WHEN status = '` + string(StatusAbandoned) + `' THEN '` + string(StatusOpen) + `' ELSE status END
		WHERE child_session_id = ?`

	// touchWindow coalesces activity writes: at most one per session (or tree) per window. It is
	// far below the inactivity and TTL windows, so the stored times stay accurate enough.
	touchWindow = 10 * time.Second
	// maxTouchEntries bounds the coalescer; past it, entries older than the window are dropped.
	maxTouchEntries    = 4096
	treeTouchKeyPrefix = "tree:"
)

// touchCoalescer remembers when each session or tree was last written, so a busy session costs
// one handoff transaction per window instead of one per tool call.
type touchCoalescer struct {
	mu   sync.Mutex
	last map[string]time.Time
}

func newTouchCoalescer() *touchCoalescer {
	return &touchCoalescer{last: map[string]time.Time{}}
}

// Touch records MCP activity for a child session: its last_activity_at, and its handoff's and
// tree's last_access_at, coalesced to one write per session per 10s. An abandoned child goes
// back to open (FI-3) and the tree's collect waiters are woken. Sessions outside a tree, and
// roots, cost one in-memory lookup.
func (s *realService) Touch(sid SessionID) {
	e, ok := s.trees.lookup(sid)
	if !ok || !e.isChild {
		return
	}
	now := nowFunc()
	if !s.touches.due(string(sid), now) {
		return
	}
	var revived bool
	err := db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		revived, err = touchChildTx(tx, sid, e, sqlTime(now))
		return err
	})
	if err != nil {
		s.touches.forget(string(sid))
		s.logger.Warn("Failed to record handoff child activity", sid.Attr(), e.tree.Attr(), "error", err)
		return
	}
	if revived {
		s.waiters.notify(e.tree)
		s.logger.Info("Revived abandoned handoff child", sid.Attr(), e.handoff.Attr(), e.tree.Attr())
	}
}

// touchTrees records a read of trees (collect, status) as tree access for the TTL (RQ-1),
// coalesced like Touch. Failures are logged: the read itself succeeded.
func (s *realService) touchTrees(trees []TreeID) {
	now := nowFunc()
	var due []TreeID
	for _, tree := range trees {
		if s.touches.due(treeTouchKeyPrefix+string(tree), now) {
			due = append(due, tree)
		}
	}
	if len(due) == 0 {
		return
	}
	err := db.HandoffTx(func(tx *sql.Tx) error {
		for _, tree := range due {
			if _, err := tx.Exec(touchTreeQuery, sqlTime(now), string(tree)); err != nil {
				return errs.WrapMessage("failed to touch handoff tree", err, "tree", string(tree))
			}
		}
		return nil
	})
	if err != nil {
		for _, tree := range due {
			s.touches.forget(treeTouchKeyPrefix + string(tree))
		}
		s.logger.Warn("Failed to record handoff tree access", "trees", len(due), "error", err)
	}
}

// touchChildTx stamps the child's activity and its handoff's and tree's access, reopening an
// abandoned child, and reports whether it did. A child whose tree was flushed is a no-op.
func touchChildTx(tx *sql.Tx, sid SessionID, e treeEntry, now string) (bool, error) {
	var status Status
	err := tx.QueryRow(selectChildStatusQuery, string(sid)).Scan(&status)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, errs.WrapMessage("failed to read handoff child status", err, "session", string(sid))
	}
	if _, err := tx.Exec(touchChildQuery, now, string(sid)); err != nil {
		return false, errs.WrapMessage("failed to touch handoff child", err, "session", string(sid))
	}
	if _, err := tx.Exec(touchHandoffQuery, now, string(e.handoff)); err != nil {
		return false, errs.WrapMessage("failed to touch handoff", err, "handoff", string(e.handoff))
	}
	if _, err := tx.Exec(touchTreeQuery, now, string(e.tree)); err != nil {
		return false, errs.WrapMessage("failed to touch handoff tree", err, "tree", string(e.tree))
	}
	return status == StatusAbandoned, nil
}

// due reports whether key's write is due at now, and if so records now as its last write.
func (c *touchCoalescer) due(key string, now time.Time) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if last, ok := c.last[key]; ok && now.Sub(last) < touchWindow && !now.Before(last) {
		return false
	}
	if len(c.last) >= maxTouchEntries {
		for k, t := range c.last {
			if now.Sub(t) >= touchWindow {
				delete(c.last, k)
			}
		}
	}
	c.last[key] = now
	return true
}

// forget drops key so its next call writes, after a failed write.
func (c *touchCoalescer) forget(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.last, key)
}
