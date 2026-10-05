package handoff

import (
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// DefaultTreeViewLimit and MaxTreeViewLimit bound how many trees TreeViews returns.
	DefaultTreeViewLimit = 20
	MaxTreeViewLimit     = 100

	// newestTreesSubquery picks the trees a TreeViews call covers; every query below filters by
	// it so the four reads agree on the same trees.
	newestTreesSubquery = `(SELECT tree_id FROM handoff_trees ORDER BY created_at DESC, tree_id LIMIT ?)`

	selectTreeViewsQuery = `SELECT tree_id, root_session_id, COALESCE(project_path, ''), created_at, last_access_at,
		tokens_used, entries_used FROM handoff_trees ORDER BY created_at DESC, tree_id LIMIT ?`
	selectHandoffViewsQuery = `SELECT ref, tree_id, parent_session_id, COALESCE(parent_child_session_id, ''), depth, mode,
		COALESCE(label, ''), created_at FROM handoffs WHERE tree_id IN ` + newestTreesSubquery + ` ORDER BY created_at, ref`
	selectChildViewsQuery = `SELECT c.child_session_id, c.handoff_ref, COALESCE(c.label, ''), c.status, h.depth, c.opened_at,
		c.last_activity_at, COALESCE(c.result_ref, ''), COALESCE(c.summary, ''), c.search_calls, c.repeat_calls,
		c.tokens_available, c.tokens_delivered
		FROM handoff_children c JOIN handoffs h ON h.ref = c.handoff_ref
		WHERE c.tree_id IN ` + newestTreesSubquery + ` ORDER BY c.opened_at, c.child_session_id`
	selectClaimCountsQuery = `SELECT tree_id, holder_session_id, COUNT(*) FROM handoff_claims
		WHERE tree_id IN ` + newestTreesSubquery + ` GROUP BY tree_id, holder_session_id`
	selectQueueCountsQuery = `SELECT tree_id, session_id, COUNT(*) FROM handoff_claim_queue
		WHERE tree_id IN ` + newestTreesSubquery + ` GROUP BY tree_id, session_id`
)

// TreeView is one handoff tree for the dashboard tree view (OB-4): its usage against the caps,
// expiry, claim totals, and its handoffs with their children.
type TreeView struct {
	TreeID          TreeID        `json:"tree_id"`
	RootSessionID   SessionID     `json:"root_session_id"`
	ProjectPath     string        `json:"project_path,omitempty"`
	CreatedAt       string        `json:"created_at"`
	LastAccessAt    string        `json:"last_access_at"`
	ExpiresAt       string        `json:"expires_at"`
	Expired         bool          `json:"expired"`
	TokensUsed      int           `json:"tokens_used"`
	TokensMax       int           `json:"tokens_max"`
	EntriesUsed     int           `json:"entries_used"`
	EntriesMax      int           `json:"entries_max"`
	ActiveClaims    int           `json:"active_claims"`
	QueuedClaims    int           `json:"queued_claims"`
	SearchCalls     int           `json:"search_calls"`
	RepeatCalls     int           `json:"repeat_calls"`
	RepeatRate      float64       `json:"repeat_rate"`
	TokensDelivered int           `json:"tokens_delivered"`
	TokensSaved     int           `json:"tokens_saved"`
	Handoffs        []HandoffView `json:"handoffs"`
}

// HandoffView is one handoff in a TreeView. ParentChildSessionID is set for a nested handoff:
// the child session that created it.
type HandoffView struct {
	Ref                  HandoffRef  `json:"handoff"`
	Label                string      `json:"label,omitempty"`
	Mode                 Mode        `json:"mode"`
	Depth                int         `json:"depth"`
	ParentSessionID      SessionID   `json:"parent_session_id"`
	ParentChildSessionID SessionID   `json:"parent_child_session_id,omitempty"`
	CreatedAt            string      `json:"created_at"`
	Children             []ChildView `json:"children"`
}

// ChildView is one child session in a HandoffView, with its OB-1 and OB-2 numbers.
type ChildView struct {
	SessionID       SessionID `json:"session_id"`
	Label           string    `json:"label,omitempty"`
	Status          Status    `json:"status"`
	Depth           int       `json:"depth"`
	OpenedAt        string    `json:"opened_at"`
	LastActivityAt  string    `json:"last_activity_at"`
	ResultRef       string    `json:"result_ref,omitempty"`
	Summary         string    `json:"summary,omitempty"`
	SearchCalls     int       `json:"search_calls"`
	RepeatCalls     int       `json:"repeat_calls"`
	RepeatRate      float64   `json:"repeat_rate"`
	TokensAvailable int       `json:"tokens_available"`
	TokensDelivered int       `json:"tokens_delivered"`
	TokensSaved     int       `json:"tokens_saved"`
	ActiveClaims    int       `json:"active_claims"`
	QueuedClaims    int       `json:"queued_claims"`
}

// claimKey counts claims per session within a tree.
type claimKey struct {
	tree TreeID
	sid  SessionID
}

// TreeViews returns the newest limit trees (DefaultTreeViewLimit when limit ≤ 0, at most
// MaxTreeViewLimit), newest first, read straight from context.db. It doesn't touch the trees:
// looking at them on the dashboard doesn't extend their TTL.
func TreeViews(limit int) ([]TreeView, error) {
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	if limit <= 0 {
		limit = DefaultTreeViewLimit
	}
	limit = min(limit, MaxTreeViewLimit)
	lim := LoadLimits()
	trees, index, err := readTreeViews(limit, lim)
	if err != nil {
		return nil, err
	}
	if len(trees) == 0 {
		return trees, nil
	}
	held, err := readClaimCounts(selectClaimCountsQuery, limit)
	if err != nil {
		return nil, err
	}
	queued, err := readClaimCounts(selectQueueCountsQuery, limit)
	if err != nil {
		return nil, err
	}
	for k, n := range held {
		if i, ok := index[k.tree]; ok {
			trees[i].ActiveClaims += n
		}
	}
	for k, n := range queued {
		if i, ok := index[k.tree]; ok {
			trees[i].QueuedClaims += n
		}
	}
	handoffAt, err := readHandoffViews(limit, trees, index)
	if err != nil {
		return nil, err
	}
	if err := readChildViews(limit, trees, handoffAt, held, queued); err != nil {
		return nil, err
	}
	for i := range trees {
		trees[i].RepeatRate = repeatRate(trees[i].SearchCalls, trees[i].RepeatCalls)
	}
	return trees, nil
}

func readTreeViews(limit int, lim Limits) ([]TreeView, map[TreeID]int, error) {
	rows, err := db.ContextDB.Query(selectTreeViewsQuery, limit)
	if err != nil {
		return nil, nil, errs.WrapMessage("failed to list handoff trees", err)
	}
	defer rows.Close()
	trees := []TreeView{}
	index := map[TreeID]int{}
	now := nowFunc()
	for rows.Next() {
		v := TreeView{TokensMax: lim.TreeMaxTokens, EntriesMax: lim.TreeMaxEntries, Handoffs: []HandoffView{}}
		if err := rows.Scan(&v.TreeID, &v.RootSessionID, &v.ProjectPath, &v.CreatedAt, &v.LastAccessAt, &v.TokensUsed, &v.EntriesUsed); err != nil {
			return nil, nil, errs.WrapMessage("failed to read handoff tree", err)
		}
		if at, err := time.ParseInLocation(time.DateTime, v.LastAccessAt, time.UTC); err == nil {
			exp := at.Add(lim.TTL())
			v.ExpiresAt, v.Expired = sqlTime(exp), !now.Before(exp)
		}
		index[v.TreeID] = len(trees)
		trees = append(trees, v)
	}
	return trees, index, rows.Err()
}

// readHandoffViews attaches each tree's handoffs and returns where each handoff landed.
func readHandoffViews(limit int, trees []TreeView, index map[TreeID]int) (map[HandoffRef][2]int, error) {
	rows, err := db.ContextDB.Query(selectHandoffViewsQuery, limit)
	if err != nil {
		return nil, errs.WrapMessage("failed to list handoffs", err)
	}
	defer rows.Close()
	at := map[HandoffRef][2]int{}
	for rows.Next() {
		var tree TreeID
		h := HandoffView{Children: []ChildView{}}
		if err := rows.Scan(&h.Ref, &tree, &h.ParentSessionID, &h.ParentChildSessionID, &h.Depth, &h.Mode, &h.Label, &h.CreatedAt); err != nil {
			return nil, errs.WrapMessage("failed to read handoff", err)
		}
		i, ok := index[tree]
		if !ok {
			continue
		}
		at[h.Ref] = [2]int{i, len(trees[i].Handoffs)}
		trees[i].Handoffs = append(trees[i].Handoffs, h)
	}
	return at, rows.Err()
}

// readChildViews attaches each handoff's children and rolls their counters up into the tree.
func readChildViews(limit int, trees []TreeView, handoffAt map[HandoffRef][2]int, held, queued map[claimKey]int) error {
	rows, err := db.ContextDB.Query(selectChildViewsQuery, limit)
	if err != nil {
		return errs.WrapMessage("failed to list handoff children", err)
	}
	defer rows.Close()
	for rows.Next() {
		var ref HandoffRef
		var c ChildView
		if err := rows.Scan(&c.SessionID, &ref, &c.Label, &c.Status, &c.Depth, &c.OpenedAt, &c.LastActivityAt, &c.ResultRef,
			&c.Summary, &c.SearchCalls, &c.RepeatCalls, &c.TokensAvailable, &c.TokensDelivered); err != nil {
			return errs.WrapMessage("failed to read handoff child", err)
		}
		pos, ok := handoffAt[ref]
		if !ok {
			continue
		}
		tree := &trees[pos[0]]
		c.RepeatRate = repeatRate(c.SearchCalls, c.RepeatCalls)
		c.TokensSaved = max(0, c.TokensAvailable-c.TokensDelivered)
		c.ActiveClaims, c.QueuedClaims = held[claimKey{tree.TreeID, c.SessionID}], queued[claimKey{tree.TreeID, c.SessionID}]
		tree.SearchCalls += c.SearchCalls
		tree.RepeatCalls += c.RepeatCalls
		tree.TokensDelivered += c.TokensDelivered
		tree.TokensSaved += c.TokensSaved
		h := &tree.Handoffs[pos[1]]
		h.Children = append(h.Children, c)
	}
	return rows.Err()
}

func readClaimCounts(q string, limit int) (map[claimKey]int, error) {
	rows, err := db.ContextDB.Query(q, limit)
	if err != nil {
		return nil, errs.WrapMessage("failed to count handoff claims", err)
	}
	defer rows.Close()
	out := map[claimKey]int{}
	for rows.Next() {
		var k claimKey
		var n int
		if err := rows.Scan(&k.tree, &k.sid, &n); err != nil {
			return nil, errs.WrapMessage("failed to read handoff claim count", err)
		}
		out[k] = n
	}
	return out, rows.Err()
}
