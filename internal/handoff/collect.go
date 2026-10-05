package handoff

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	handoffColumns            = `ref, tree_id, COALESCE(label, ''), mode, depth, created_at`
	selectHandoffByRefQuery   = `SELECT ` + handoffColumns + ` FROM handoffs WHERE ref = ?`
	selectParentHandoffsQuery = `SELECT ` + handoffColumns + ` FROM handoffs WHERE parent_session_id = ?
		ORDER BY created_at DESC, rowid DESC`
	selectHandoffsByTreeQuery = `SELECT ` + handoffColumns + ` FROM handoffs WHERE tree_id = ? ORDER BY created_at, rowid`
	// A child's label falls back to its handoff's. Claim and note counts are subqueries: every
	// table is in context.db.
	selectHandoffChildrenQuery = `SELECT c.child_session_id, COALESCE(NULLIF(c.label, ''), h.label, ''), h.depth, c.status,
		COALESCE(c.result_ref, ''), COALESCE(c.summary, ''), c.summary_truncated, c.last_activity_at,
		(SELECT COUNT(*) FROM handoff_claims cl WHERE cl.tree_id = c.tree_id AND cl.holder_session_id = c.child_session_id),
		(SELECT COUNT(*) FROM context_notes n WHERE n.session_id = c.child_session_id),
		COALESCE((SELECT n.metadata_json FROM context_notes n WHERE n.ref = c.result_ref), '')
		FROM handoff_children c JOIN handoffs h ON h.ref = c.handoff_ref
		WHERE c.handoff_ref = ? ORDER BY c.rowid`
	selectChildStatusCountsQuery = `SELECT status, COUNT(*) FROM handoff_children WHERE handoff_ref = ? GROUP BY status`
	selectTreeQuery              = `SELECT root_session_id, COALESCE(project_path, ''), created_at, last_access_at,
		tokens_used, entries_used FROM handoff_trees WHERE tree_id = ?`
	selectTreeExistsQuery    = `SELECT 1 FROM handoff_trees WHERE tree_id = ?`
	selectRootTreesQuery     = `SELECT tree_id FROM handoff_trees WHERE root_session_id = ? ORDER BY created_at DESC, rowid DESC`
	selectChildTreeQuery     = `SELECT tree_id FROM handoff_children WHERE child_session_id = ?`
	countTreeClaimsQuery     = `SELECT COUNT(*) FROM handoff_claims WHERE tree_id = ?`
	countTreeClaimQueueQuery = `SELECT COUNT(*) FROM handoff_claim_queue WHERE tree_id = ?`

	// maxWaitSeconds caps a collect long-poll (FI-6).
	maxWaitSeconds = 60
	// collectEntryTokens is a child entry's size without its summary: ids, status, times, and
	// counts. The default collect budget fits MaxChildren entries with full summaries (FI-1).
	collectEntryTokens = 80
)

// fanInHandoff is one handoffs row as collect, list, and status read it.
type fanInHandoff struct {
	ref       HandoffRef
	tree      TreeID
	label     string
	mode      Mode
	depth     int
	createdAt string
}

// resultMeta is the part of a result note's metadata collect echoes (RT-8).
type resultMeta struct {
	ChangedFiles  []string `json:"changed_files"`
	OpenQuestions []string `json:"open_questions"`
}

// Collect returns the children of one handoff, or of every handoff the session created, with
// their results and activity (FI-1, FI-4). Recursive adds the handoffs those children created,
// depth first (FI-5). WaitSeconds long-polls while any child is open, returning on the first
// status change in the trees involved or at the timeout (FI-6).
func (s *realService) Collect(ctx context.Context, req CollectRequest) (*CollectResponse, error) {
	roots, err := collectRoots(req)
	if err != nil {
		return nil, err
	}
	trees := uniqueTrees(roots)
	s.touchTrees(trees)
	budget := req.TokenBudget
	if budget <= 0 {
		lim := LoadLimits()
		budget = lim.MaxChildren * (lim.SummaryMaxTokens + collectEntryTokens)
	}
	wait := time.Duration(min(max(req.WaitSeconds, 0), maxWaitSeconds)) * time.Second
	var wake <-chan struct{}
	if wait > 0 {
		// Taken before the read, so a change between the read and the wait still wakes us.
		var stop func()
		wake, stop = s.waitAny(trees)
		defer stop()
	}
	res, anyOpen, err := collectChildren(roots, req.Recursive, budget)
	if err != nil || wait == 0 || !anyOpen {
		return res, err
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, errs.WrapMessage("collect wait canceled", ctx.Err())
	case <-timer.C:
	case <-wake:
	}
	res, _, err = collectChildren(roots, req.Recursive, budget)
	if err != nil {
		return nil, err
	}
	res.Waited = true
	return res, nil
}

// List returns the handoffs a session created, newest first, with per-status child counts.
// It needs only the session id, so a parent can recover refs lost to compaction (FI-2).
func (s *realService) List(ctx context.Context, req ListRequest) (*ListResponse, error) {
	if req.SessionID == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "session_id required")
	}
	rows, err := handoffsByParent(req.SessionID)
	if err != nil {
		return nil, err
	}
	out := &ListResponse{Handoffs: make([]HandoffSummary, 0, len(rows))}
	for _, h := range rows {
		sum, err := summarizeHandoff(h)
		if err != nil {
			return nil, err
		}
		out.Handoffs = append(out.Handoffs, sum)
	}
	return out, nil
}

// Status returns a compact view of the tree named by tree id, handoff ref, or session (a
// child's tree, or a root's newest tree): usage against the caps, expiry, handoffs, and claims.
func (s *realService) Status(ctx context.Context, req StatusRequest) (*StatusResponse, error) {
	trees, err := resolveTrees(req.TreeID, req.Handoff, req.SessionID, false)
	if err != nil {
		return nil, err
	}
	tree := trees[0]
	out := &StatusResponse{TreeID: tree}
	err = db.ContextDB.QueryRow(selectTreeQuery, string(tree)).Scan(&out.RootSessionID, &out.ProjectPath, &out.CreatedAt,
		&out.LastAccessAt, &out.TokensUsed, &out.EntriesUsed)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errs.NewCode(CodeHandoffNotFound, "handoff tree not found", "tree", string(tree))
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to read handoff tree", err, "tree", string(tree))
	}
	s.touchTrees([]TreeID{tree})
	lim := LoadLimits()
	out.TokensMax, out.EntriesMax = lim.TreeMaxTokens, lim.TreeMaxEntries
	if at, err := time.ParseInLocation(time.DateTime, out.LastAccessAt, time.UTC); err == nil {
		out.ExpiresAt = sqlTime(at.Add(lim.TTL()))
	}
	rows, err := queryHandoffs(selectHandoffsByTreeQuery, string(tree))
	if err != nil {
		return nil, err
	}
	out.Handoffs = make([]HandoffSummary, 0, len(rows))
	for _, h := range rows {
		sum, err := summarizeHandoff(h)
		if err != nil {
			return nil, err
		}
		out.Handoffs = append(out.Handoffs, sum)
	}
	if err := db.ContextDB.QueryRow(countTreeClaimsQuery, string(tree)).Scan(&out.ActiveClaims); err != nil {
		return nil, errs.WrapMessage("failed to count handoff claims", err, "tree", string(tree))
	}
	if err := db.ContextDB.QueryRow(countTreeClaimQueueQuery, string(tree)).Scan(&out.QueuedClaims); err != nil {
		return nil, errs.WrapMessage("failed to count queued handoff claims", err, "tree", string(tree))
	}
	return out, nil
}

// Flush deletes a tree and everything it owns (RQ-3), named by tree id, by any of its handoff
// refs, or by its root session. A root session can have rooted several trees; flushing by
// session deletes them all, and the response sums their counts under the newest tree's id.
// Only the root may flush by session id; a child names the tree by its handoff ref.
func (s *realService) Flush(ctx context.Context, req FlushRequest) (*FlushResponse, error) {
	trees, err := resolveTrees(req.TreeID, req.Handoff, req.SessionID, true)
	if err != nil {
		return nil, err
	}
	if err := requireTree(trees[0]); err != nil {
		return nil, err
	}
	total := &FlushResponse{TreeID: trees[0]}
	for _, tree := range trees {
		res, err := s.flushTree(tree, false)
		if err != nil {
			return nil, err
		}
		total.Handoffs += res.Handoffs
		total.Children += res.Children
		total.NotesDeleted += res.NotesDeleted
		total.MemoryDeleted += res.MemoryDeleted
	}
	return total, nil
}

// waitAny returns a channel that is closed when any of trees is notified, and a stop func
// that releases its watchers.
func (s *realService) waitAny(trees []TreeID) (<-chan struct{}, func()) {
	if len(trees) == 1 {
		return s.waiters.wait(trees[0]), func() {}
	}
	woke, done := make(chan struct{}), make(chan struct{})
	var once sync.Once
	for _, tree := range trees {
		ch := s.waiters.wait(tree)
		go func() {
			select {
			case <-ch:
				once.Do(func() { close(woke) })
			case <-done:
			}
		}()
	}
	return woke, func() { close(done) }
}

// collectRoots resolves a collect request to the handoffs whose children it lists.
func collectRoots(req CollectRequest) ([]fanInHandoff, error) {
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	switch {
	case req.Handoff != "":
		if _, err := ParseHandoffRef(string(req.Handoff)); err != nil {
			return nil, err
		}
		h, err := handoffByRef(req.Handoff)
		if err != nil {
			return nil, err
		}
		return []fanInHandoff{h}, nil
	case req.SessionID != "":
		rows, err := handoffsByParent(req.SessionID)
		if err != nil {
			return nil, err
		}
		// Oldest first, so children read in the order they were handed off.
		for i, j := 0, len(rows)-1; i < j; i, j = i+1, j-1 {
			rows[i], rows[j] = rows[j], rows[i]
		}
		return rows, nil
	}
	return nil, errs.NewCode(errs.CodeInvalidInput, "handoff or session_id required")
}

// collectChildren lists the children of roots (and, recursively, of the handoffs they created)
// within budget tokens, and reports whether any child it read is still open. Every child is
// read, so anyOpen holds even when the response is truncated.
func collectChildren(roots []fanInHandoff, recursive bool, budget int) (*CollectResponse, bool, error) {
	res := &CollectResponse{Children: []ChildResult{}}
	anyOpen := false
	seen := map[HandoffRef]bool{}
	var visit func(h fanInHandoff) error
	visit = func(h fanInHandoff) error {
		if seen[h.ref] {
			return nil
		}
		seen[h.ref] = true
		children, err := handoffChildren(h.ref)
		if err != nil {
			return err
		}
		for _, c := range children {
			anyOpen = anyOpen || c.Status == StatusOpen
			cost := childTokens(c)
			if res.Truncated || res.TokensUsed+cost > budget {
				res.Truncated = true
			} else {
				res.Children = append(res.Children, c)
				res.TokensUsed += cost
			}
			if !recursive {
				continue
			}
			nested, err := handoffsByParent(c.SessionID)
			if err != nil {
				return err
			}
			for i := len(nested) - 1; i >= 0; i-- {
				if err := visit(nested[i]); err != nil {
					return err
				}
			}
		}
		return nil
	}
	for _, h := range roots {
		if err := visit(h); err != nil {
			return nil, false, err
		}
	}
	return res, anyOpen, nil
}

func handoffChildren(ref HandoffRef) ([]ChildResult, error) {
	rows, err := db.ContextDB.Query(selectHandoffChildrenQuery, string(ref))
	if err != nil {
		return nil, errs.WrapMessage("failed to list handoff children", err, "handoff", string(ref))
	}
	defer rows.Close()
	var out []ChildResult
	for rows.Next() {
		c := ChildResult{Handoff: ref}
		var meta string
		if err := rows.Scan(&c.SessionID, &c.Label, &c.Depth, &c.Status, &c.ResultRef, &c.Summary, &c.SummaryTruncated,
			&c.LastActivityAt, &c.ActiveClaims, &c.NoteCount, &meta); err != nil {
			return nil, errs.WrapMessage("failed to read handoff child", err, "handoff", string(ref))
		}
		if meta != "" {
			var m resultMeta
			if json.Unmarshal([]byte(meta), &m) == nil {
				c.ChangedFiles, c.OpenQuestions = m.ChangedFiles, m.OpenQuestions
			}
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// childTokens is the entry's size as the parent reads it.
func childTokens(c ChildResult) int {
	b, _ := json.Marshal(c)
	return db.EstimateTokens(string(b))
}

func summarizeHandoff(h fanInHandoff) (HandoffSummary, error) {
	sum := HandoffSummary{Ref: h.ref, TreeID: h.tree, Label: h.label, Mode: h.mode, Depth: h.depth, CreatedAt: h.createdAt}
	rows, err := db.ContextDB.Query(selectChildStatusCountsQuery, string(h.ref))
	if err != nil {
		return sum, errs.WrapMessage("failed to count handoff children", err, "handoff", string(h.ref))
	}
	defer rows.Close()
	for rows.Next() {
		var st Status
		var n int
		if err := rows.Scan(&st, &n); err != nil {
			return sum, errs.WrapMessage("failed to read handoff child count", err, "handoff", string(h.ref))
		}
		if sum.StatusCounts == nil {
			sum.StatusCounts = map[Status]int{}
		}
		sum.StatusCounts[st] = n
		sum.Children += n
	}
	return sum, rows.Err()
}

func handoffByRef(ref HandoffRef) (fanInHandoff, error) {
	rows, err := queryHandoffs(selectHandoffByRefQuery, string(ref))
	if err != nil {
		return fanInHandoff{}, err
	}
	if len(rows) == 0 {
		return fanInHandoff{}, errs.NewCode(CodeHandoffNotFound, "handoff not found", "handoff", string(ref))
	}
	return rows[0], nil
}

func handoffsByParent(sid SessionID) ([]fanInHandoff, error) {
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	return queryHandoffs(selectParentHandoffsQuery, string(sid))
}

func queryHandoffs(q string, arg string) ([]fanInHandoff, error) {
	rows, err := db.ContextDB.Query(q, arg)
	if err != nil {
		return nil, errs.WrapMessage("failed to list handoffs", err)
	}
	defer rows.Close()
	var out []fanInHandoff
	for rows.Next() {
		var h fanInHandoff
		if err := rows.Scan(&h.ref, &h.tree, &h.label, &h.mode, &h.depth, &h.createdAt); err != nil {
			return nil, errs.WrapMessage("failed to read handoff", err)
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// resolveTrees maps a tree id, handoff ref, or session id (checked in that order) to trees. A
// session resolves to the trees it roots, newest first; unless rootOnly, a child session
// resolves to its tree. The first tree is the one a single-tree caller should use.
func resolveTrees(tree TreeID, ref HandoffRef, sid SessionID, rootOnly bool) ([]TreeID, error) {
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	switch {
	case tree != "":
		if _, err := ParseTreeID(string(tree)); err != nil {
			return nil, err
		}
		return []TreeID{tree}, nil
	case ref != "":
		if _, err := ParseHandoffRef(string(ref)); err != nil {
			return nil, err
		}
		h, err := handoffByRef(ref)
		if err != nil {
			return nil, err
		}
		return []TreeID{h.tree}, nil
	case sid != "":
		return sessionTrees(sid, rootOnly)
	}
	return nil, errs.NewCode(errs.CodeInvalidInput, "tree_id, handoff, or session_id required")
}

func sessionTrees(sid SessionID, rootOnly bool) ([]TreeID, error) {
	if !rootOnly {
		var tree TreeID
		err := db.ContextDB.QueryRow(selectChildTreeQuery, string(sid)).Scan(&tree)
		if err == nil {
			return []TreeID{tree}, nil
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return nil, errs.WrapMessage("failed to read handoff child tree", err, "session", string(sid))
		}
	}
	rows, err := db.ContextDB.Query(selectRootTreesQuery, string(sid))
	if err != nil {
		return nil, errs.WrapMessage("failed to list rooted handoff trees", err, "session", string(sid))
	}
	defer rows.Close()
	var out []TreeID
	for rows.Next() {
		var tree TreeID
		if err := rows.Scan(&tree); err != nil {
			return nil, errs.WrapMessage("failed to read rooted handoff tree", err, "session", string(sid))
		}
		out = append(out, tree)
	}
	if err := rows.Err(); err != nil {
		return nil, errs.WrapMessage("failed to list rooted handoff trees", err, "session", string(sid))
	}
	if len(out) == 0 {
		return nil, errs.NewCode(CodeHandoffNotFound, "session roots no handoff tree", "session", string(sid))
	}
	return out, nil
}

func requireTree(tree TreeID) error {
	var one int
	err := db.ContextDB.QueryRow(selectTreeExistsQuery, string(tree)).Scan(&one)
	if errors.Is(err, sql.ErrNoRows) {
		return errs.NewCode(CodeHandoffNotFound, "handoff tree not found", "tree", string(tree))
	}
	if err != nil {
		return errs.WrapMessage("failed to read handoff tree", err, "tree", string(tree))
	}
	return nil
}

func uniqueTrees(rows []fanInHandoff) []TreeID {
	seen := map[TreeID]bool{}
	var out []TreeID
	for _, h := range rows {
		if !seen[h.tree] {
			seen[h.tree] = true
			out = append(out, h.tree)
		}
	}
	return out
}
