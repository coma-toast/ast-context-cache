package handoff

import (
	"database/sql"
	"errors"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	selectChildTreeEntryQuery = `SELECT c.tree_id, c.handoff_ref, COALESCE(h.mode, '` + string(ModeFresh) + `')
		FROM handoff_children c LEFT JOIN handoffs h ON h.ref = c.handoff_ref
		WHERE c.child_session_id = ?`
	selectRootTreeQuery = `SELECT tree_id FROM handoff_trees WHERE root_session_id = ? ORDER BY created_at DESC LIMIT 1`
	// maxNegativeEntries bounds the remembered non-tree sessions; past it they are forgotten
	// and re-checked on their next call.
	maxNegativeEntries = 4096
)

// treeEntry is a session's place in a tree.
type treeEntry struct {
	tree    TreeID
	handoff HandoffRef // the handoff a child opened; empty for a root
	mode    Mode
	isChild bool
}

// treeIndex maps session ids to their tree, hydrated lazily from handoff_children and
// handoff_trees. It also remembers sessions that are in no tree, so the per-call check costs
// a map lookup for them; every write that puts a session into a tree goes through this
// process and calls put, so a remembered miss can't go stale.
type treeIndex struct {
	mu        sync.RWMutex
	entries   map[SessionID]*treeEntry // nil value: known not to be in a tree
	negatives int
}

func newTreeIndex() *treeIndex {
	return &treeIndex{entries: map[SessionID]*treeEntry{}}
}

// lookup returns sid's tree entry. A database error is not remembered, so the next call retries.
func (t *treeIndex) lookup(sid SessionID) (treeEntry, bool) {
	if sid == "" {
		return treeEntry{}, false
	}
	t.mu.RLock()
	e, known := t.entries[sid]
	t.mu.RUnlock()
	if known {
		if e == nil {
			return treeEntry{}, false
		}
		return *e, true
	}
	e, err := hydrateTreeEntry(sid)
	if err != nil {
		return treeEntry{}, false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	// A put that raced the hydration wins: it reflects a write made after our read.
	if cur, ok := t.entries[sid]; ok && cur != nil {
		return *cur, true
	}
	t.store(sid, e)
	if e == nil {
		return treeEntry{}, false
	}
	return *e, true
}

// put records sid's tree after a write that placed it there.
func (t *treeIndex) put(sid SessionID, e treeEntry) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.store(sid, &e)
}

// forgetTree drops every session of tree, so they rehydrate (to "no tree") on their next call.
func (t *treeIndex) forgetTree(tree TreeID) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for sid, e := range t.entries {
		if e != nil && e.tree == tree {
			delete(t.entries, sid)
		}
	}
}

// store must be called with mu held.
func (t *treeIndex) store(sid SessionID, e *treeEntry) {
	prev, existed := t.entries[sid]
	if existed && prev == nil {
		t.negatives--
	}
	if e != nil {
		t.entries[sid] = e
		return
	}
	if t.negatives >= maxNegativeEntries {
		for k, v := range t.entries {
			if v == nil {
				delete(t.entries, k)
			}
		}
		t.negatives = 0
	}
	t.entries[sid] = nil
	t.negatives++
}

// hydrateTreeEntry reads sid's tree from the database: as a child first, then as a root. It
// returns nil, nil for a session in no tree.
func hydrateTreeEntry(sid SessionID) (*treeEntry, error) {
	conn := db.ContextDB
	if conn == nil {
		return nil, errNoContextDB
	}
	var e treeEntry
	var mode string
	err := conn.QueryRow(selectChildTreeEntryQuery, string(sid)).Scan(&e.tree, &e.handoff, &mode)
	if err == nil {
		e.mode, e.isChild = Mode(mode), true
		return &e, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return nil, err
	}
	err = conn.QueryRow(selectRootTreeQuery, string(sid)).Scan(&e.tree)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &e, nil
}
