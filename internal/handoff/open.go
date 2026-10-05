package handoff

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/repokey"
)

const (
	// pagingReserveTokens is room kept for "truncated" and "next", which are set only after the
	// item that didn't fit is taken back out.
	pagingReserveTokens = 16
	// headlineMaxBytes caps a scratchpad headline in the open digest.
	headlineMaxBytes = 100
	// digestLatestEntries is how many latest headlines and dead ends the scratchpad digest shows.
	digestLatestEntries = 3
	// digestMaxClaims caps the active claims the scratchpad digest shows.
	digestMaxClaims = 10

	selectHandoffQuery = `SELECT h.tree_id, h.parent_session_id, h.depth, h.mode, COALESCE(h.label, ''), h.brief,
		COALESCE(h.project_path, ''), h.child_count, t.last_access_at
		FROM handoffs h JOIN handoff_trees t ON t.tree_id = h.tree_id WHERE h.ref = ?`
	selectChildQuery = `SELECT handoff_ref, status, COALESCE(project_path, ''), tokens_available, tokens_delivered
		FROM handoff_children WHERE child_session_id = ?`
	selectAvailableTokensQuery = `SELECT COALESCE(SUM(token_est), 0) FROM handoff_snapshot_items
		WHERE handoff_ref = ? AND section IN ('` + string(SectionNote) + `', '` + string(SectionPointer) + `', '` + string(SectionTrail) + `')`
	insertChildQuery = `INSERT INTO handoff_children (child_session_id, handoff_ref, tree_id, label, status, project_path,
		opened_at, last_activity_at, tokens_available) VALUES (?, ?, ?, ?, '` + string(StatusOpen) + `', ?, ?, ?, ?)`
	bumpChildCountQuery = `UPDATE handoffs SET child_count = ?, last_access_at = ? WHERE ref = ?`
	touchHandoffQuery   = `UPDATE handoffs SET last_access_at = ? WHERE ref = ?`
	resumeChildQuery    = `UPDATE handoff_children SET last_activity_at = ?,
		status = CASE WHEN status = '` + string(StatusAbandoned) + `' THEN '` + string(StatusOpen) + `' ELSE status END,
		project_path = COALESCE(?, project_path) WHERE child_session_id = ?`
	addDeliveredQuery   = `UPDATE handoff_children SET tokens_delivered = tokens_delivered + ? WHERE child_session_id = ?`
	selectDeliveryQuery = `SELECT tokens_available, tokens_delivered FROM handoff_children WHERE child_session_id = ?`
	touchChildTreeQuery = `UPDATE handoff_trees SET last_access_at = ?
		WHERE tree_id = (SELECT tree_id FROM handoff_children WHERE child_session_id = ?)`

	selectScratchpadCountsQuery = `SELECT type, COUNT(*) FROM scratchpad_entries WHERE tree_id = ? AND retracted_at IS NULL GROUP BY type`
	selectScratchpadLatestQuery = `SELECT id, type, author_session_id, text FROM scratchpad_entries
		WHERE tree_id = ? AND retracted_at IS NULL AND type != '` + string(EntryTypeTrail) + `' ORDER BY id DESC LIMIT ?`
	selectScratchpadDeadEndsQuery = `SELECT id, type, author_session_id, text FROM scratchpad_entries
		WHERE tree_id = ? AND retracted_at IS NULL AND (type = '` + string(EntryTypeDeadEnd) + `'
			OR (type = '` + string(EntryTypeTrail) + `' AND json_extract(refs_json, '$.zero_hit') = 1))
		ORDER BY id DESC LIMIT ?`
	selectTreeClaimsQuery = `SELECT c.key, c.holder_session_id, COALESCE(hc.label, ''), COALESCE(c.reason, ''), c.granted_at
		FROM handoff_claims c LEFT JOIN handoff_children hc ON hc.child_session_id = c.holder_session_id
		WHERE c.tree_id = ? ORDER BY c.granted_at, c.key LIMIT ?`
	selectTreeClaimQueueQuery = `SELECT key, session_id, COALESCE(reason, ''), enqueued_at FROM handoff_claim_queue
		WHERE tree_id = ? ORDER BY key, id`
)

// digestSections are the pageable open-digest sections, in priority order (OP-3).
var digestSections = []Section{SectionPointer, SectionNote, SectionMemory, SectionTrail}

// handoffRow is a handoff joined with its tree's last access.
type handoffRow struct {
	ref        HandoffRef
	tree       TreeID
	parent     SessionID
	depth      int
	mode       Mode
	label      string
	brief      string
	project    string
	childCount int
	treeAccess string
}

// childRow is a child session's handoff linkage and OB-2 totals.
type childRow struct {
	handoff   HandoffRef
	status    Status
	project   string
	available int
	delivered int
}

// rowQuerier is a *sql.DB or *sql.Tx.
type rowQuerier interface {
	QueryRow(query string, args ...any) *sql.Row
}

// Open mints a child session for a handoff, or resumes the child named by req.SessionID, and
// returns the compact digest (OP-1–OP-3, OP-5, OP-11, HO-9).
func (s *realService) Open(ctx context.Context, req OpenRequest) (*OpenResponse, error) {
	ref, err := ParseHandoffRef(strings.TrimSpace(string(req.Handoff)))
	if err != nil {
		return nil, err
	}
	sid := SessionID(strings.TrimSpace(string(req.SessionID)))
	if req.Next != nil && sid == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "session_id required to page an open digest", "handoff", string(ref))
	}
	l := LoadLimits()
	project := projectlinks.NormalizePath(req.ProjectPath)
	var h *handoffRow
	resumed := sid != ""
	err = db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		if h, err = loadLiveHandoff(tx, ref, l); err != nil {
			return err
		}
		if resumed {
			return resumeChildTx(tx, h, sid, project)
		}
		sid, err = mintChildTx(tx, h, project, l)
		return err
	})
	if err != nil {
		return nil, err
	}
	if !resumed {
		s.seedChild(h, sid, project)
	}
	s.trees.put(sid, treeEntry{tree: h.tree, handoff: ref, mode: h.mode, isChild: true})
	budget := req.TokenBudget
	if budget <= 0 {
		budget = l.OpenBudgetTokens
	}
	resp, err := s.openDigest(h, sid, resumed, budget, req.Next)
	if err != nil {
		return nil, err
	}
	if resp.TokensAvailable, resp.TokensDelivered, err = addDelivered(sid, resp.TokensUsed); err != nil {
		return nil, err
	}
	if resumed {
		s.logger.Info("Resumed handoff child", ref.Attr(), h.tree.Attr(), sid.Attr())
	} else {
		s.logger.Info("Opened handoff", ref.Attr(), h.tree.Attr(), sid.Attr(), "mode", string(h.mode))
	}
	return resp, nil
}

// loadLiveHandoff reads ref and fails with CodeHandoffNotFound or, past the tree TTL,
// CodeHandoffExpired.
func loadLiveHandoff(q rowQuerier, ref HandoffRef, l Limits) (*handoffRow, error) {
	h := &handoffRow{ref: ref}
	err := q.QueryRow(selectHandoffQuery, string(ref)).Scan(&h.tree, &h.parent, &h.depth, &h.mode, &h.label, &h.brief,
		&h.project, &h.childCount, &h.treeAccess)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errs.NewCode(CodeHandoffNotFound, "handoff not found", "handoff", string(ref))
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to read handoff", err, "handoff", string(ref))
	}
	if h.treeAccess < sqlTime(nowFunc().Add(-l.TTL())) {
		return nil, errs.NewCode(CodeHandoffExpired, "handoff expired", "handoff", string(ref), "tree", string(h.tree))
	}
	return h, nil
}

// loadChild reads sid's child row and fails with CodeHandoffNotFound unless it is a child of ref.
func loadChild(q rowQuerier, ref HandoffRef, sid SessionID) (*childRow, error) {
	c := &childRow{}
	err := q.QueryRow(selectChildQuery, string(sid)).Scan(&c.handoff, &c.status, &c.project, &c.available, &c.delivered)
	if errors.Is(err, sql.ErrNoRows) || (err == nil && c.handoff != ref) {
		return nil, errs.NewCode(CodeHandoffNotFound, "no child session of this handoff", "handoff", string(ref), "session_id", string(sid))
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to read handoff child", err, "session_id", string(sid))
	}
	return c, nil
}

// mintChildTx registers the handoff's next child, within handoff_max_children (HO-9).
func mintChildTx(tx *sql.Tx, h *handoffRow, project string, l Limits) (SessionID, error) {
	if h.childCount >= l.MaxChildren {
		return "", errs.NewCode(CodeHandoffChildrenExceeded, "handoff children exceeded", "handoff", string(h.ref),
			"children", h.childCount, "max_children", l.MaxChildren)
	}
	var available int
	if err := tx.QueryRow(selectAvailableTokensQuery, string(h.ref)).Scan(&available); err != nil {
		return "", errs.WrapMessage("failed to total handoff snapshot", err, "handoff", string(h.ref))
	}
	n := h.childCount + 1
	sid := ChildSessionID(h.ref, n)
	if project == "" {
		project = h.project
	}
	ts := sqlTime(nowFunc())
	if _, err := tx.Exec(insertChildQuery, string(sid), string(h.ref), string(h.tree), nullIfEmpty(h.label), nullIfEmpty(project),
		ts, ts, available); err != nil {
		return "", errs.WrapMessage("failed to insert handoff child", err, "session_id", string(sid))
	}
	if _, err := tx.Exec(bumpChildCountQuery, n, ts, string(h.ref)); err != nil {
		return "", errs.WrapMessage("failed to count handoff child", err, "handoff", string(h.ref))
	}
	if _, err := tx.Exec(touchTreeQuery, ts, string(h.tree)); err != nil {
		return "", errs.WrapMessage("failed to touch handoff tree", err, "tree", string(h.tree))
	}
	h.childCount = n
	return sid, nil
}

// resumeChildTx revives an existing child: no new id and no count bump; an abandoned child is
// open again (OP-2).
func resumeChildTx(tx *sql.Tx, h *handoffRow, sid SessionID, project string) error {
	if _, err := loadChild(tx, h.ref, sid); err != nil {
		return err
	}
	ts := sqlTime(nowFunc())
	if _, err := tx.Exec(resumeChildQuery, ts, nullIfEmpty(project), string(sid)); err != nil {
		return errs.WrapMessage("failed to resume handoff child", err, "session_id", string(sid))
	}
	if _, err := tx.Exec(touchHandoffQuery, ts, string(h.ref)); err != nil {
		return errs.WrapMessage("failed to touch handoff", err, "handoff", string(h.ref))
	}
	if _, err := tx.Exec(touchTreeQuery, ts, string(h.tree)); err != nil {
		return errs.WrapMessage("failed to touch handoff tree", err, "tree", string(h.tree))
	}
	return nil
}

// seedChild sets up a new child's session outside the handoff transaction: a fork inherits the
// parent's dedup state (OP-5), and every child gets the snapshot's memory as its own
// session-scoped entries, so recall_memory finds them (OP-11). Both are best effort: the child
// is already registered, and a failure here costs it dedup or recall, not correctness.
func (s *realService) seedChild(h *handoffRow, sid SessionID, project string) {
	if h.mode == ModeFork {
		items, err := loadSnapshotItems(h.ref, SectionManifest)
		if err != nil {
			s.logger.Warn("Failed to read handoff manifest for fork seeding", h.ref.Attr(), sid.Attr(), "error", err)
		}
		root := mappedProject(h.project, project)
		syms := make([]astcontext.ReturnedSymbol, 0, len(items))
		for _, it := range items {
			file, name, line, ok := parseDedupKey(it.key)
			if !ok {
				continue
			}
			if root != h.project && it.fileRel != "" && !filepath.IsAbs(it.fileRel) {
				file = filepath.Join(root, it.fileRel)
			}
			syms = append(syms, astcontext.ReturnedSymbol{File: file, Name: name, ProjectPath: root, StartLine: line})
		}
		astcontext.SeedReturned(string(sid), syms)
	}
	items, err := loadSnapshotItems(h.ref, SectionMemory)
	if err != nil {
		s.logger.Warn("Failed to read handoff memory for cloning", h.ref.Attr(), sid.Attr(), "error", err)
		return
	}
	for _, it := range items {
		var mc memoryContent
		if err := json.Unmarshal([]byte(it.content), &mc); err != nil {
			s.logger.Warn("Failed to decode snapshot memory", h.ref.Attr(), "ref", it.key, "error", err)
			continue
		}
		res, err := memory.Store(memory.StoreInput{
			Kind: mc.Kind, Scope: memory.ScopeSession, SessionID: string(sid), ProjectPath: mc.ProjectPath,
			Subject: mc.Subject, Predicate: mc.Predicate, Object: mc.Object, Rule: mc.Rule, SourceRef: it.key,
		})
		if err != nil {
			s.logger.Warn("Failed to clone snapshot memory into child", h.ref.Attr(), sid.Attr(), "ref", it.key, "error", err)
			continue
		}
		if s.emb != nil {
			go memory.EmbedEntry(res.Ref, string(sid), res.Line, s.emb)
		}
	}
}

// openDigest assembles the digest in priority order within budget: brief, child and mode,
// pointers, notes, memory, trail, then the scratchpad. It stops at the first item that doesn't
// fit and says where to continue (OP-3); next resumes there and omits the brief.
func (s *realService) openDigest(h *handoffRow, sid SessionID, resumed bool, budget int, next *PageCursor) (*OpenResponse, error) {
	resp := &OpenResponse{Handoff: h.ref, SessionID: sid, TreeID: h.tree, Mode: h.mode, Resumed: resumed, Label: h.label}
	if next == nil {
		resp.Brief = h.brief
	}
	// Measure with tokens_used at its largest, and room left for the paging fields, so filling
	// them in afterwards can't overflow.
	resp.TokensUsed = budget
	limit := budget - pagingReserveTokens
	start := 0
	if next != nil {
		for i, sec := range digestSections {
			if sec == next.Section {
				start = i
			}
		}
	}
	for _, sec := range digestSections[start:] {
		items, err := loadSnapshotItems(h.ref, sec)
		if err != nil {
			return nil, err
		}
		offset := 0
		if next != nil && sec == next.Section {
			offset = max(0, next.Offset)
		}
		for i := offset; i < len(items); i++ {
			undo := appendDigestItem(resp, items[i])
			if responseTokens(resp) > limit {
				undo()
				resp.Truncated, resp.Next = true, &PageCursor{Section: sec, Offset: i}
				break
			}
		}
		if resp.Truncated {
			break
		}
	}
	sp, err := scratchpadDigest(h.tree)
	if err != nil {
		return nil, err
	}
	if sp != nil {
		resp.Scratchpad = sp
		if responseTokens(resp) > limit {
			resp.Scratchpad, resp.Truncated = nil, true
		}
	}
	resp.TokensUsed = responseTokens(resp)
	return resp, nil
}

// appendDigestItem adds it to its digest section and returns how to take it back out.
func appendDigestItem(resp *OpenResponse, it snapshotItem) (undo func()) {
	switch it.section {
	case SectionPointer:
		resp.Pointers = append(resp.Pointers, PointerDigest{ID: it.id, Key: it.key, Note: it.label, Kind: it.kind})
		return func() { resp.Pointers = resp.Pointers[:len(resp.Pointers)-1] }
	case SectionNote:
		resp.Notes = append(resp.Notes, ItemDigest{ID: it.id, Ref: it.key, Label: it.label, TokenEst: it.tokenEst})
		return func() { resp.Notes = resp.Notes[:len(resp.Notes)-1] }
	case SectionMemory:
		resp.Memory = append(resp.Memory, ItemDigest{ID: it.id, Ref: it.key, Label: it.label, TokenEst: it.tokenEst})
		return func() { resp.Memory = resp.Memory[:len(resp.Memory)-1] }
	case SectionTrail:
		var tc trailContent
		_ = json.Unmarshal([]byte(it.content), &tc)
		resp.Trail = append(resp.Trail, TrailDigest{ID: it.id, Tool: tc.Tool, Query: tc.Query, Hits: tc.HitCount, ZeroHit: tc.ZeroHit})
		return func() { resp.Trail = resp.Trail[:len(resp.Trail)-1] }
	}
	return func() {}
}

// scratchpadDigest summarizes the tree's scratchpad: counts by type, the latest headlines, dead
// ends, and active claims (OP-3, SP-7, CL-8). It is nil for an empty scratchpad.
func scratchpadDigest(tree TreeID) (*ScratchpadDigest, error) {
	conn := db.ContextDB
	if conn == nil {
		return nil, errNoContextDB
	}
	d := &ScratchpadDigest{Counts: map[EntryType]int{}}
	rows, err := conn.Query(selectScratchpadCountsQuery, string(tree))
	if err != nil {
		return nil, errs.WrapMessage("failed to count scratchpad entries", err, "tree", string(tree))
	}
	for rows.Next() {
		var t EntryType
		var n int
		if rows.Scan(&t, &n) == nil {
			d.Counts[t] = n
		}
	}
	rows.Close()
	if d.Latest, err = headlines(conn, selectScratchpadLatestQuery, tree); err != nil {
		return nil, err
	}
	if d.DeadEnds, err = headlines(conn, selectScratchpadDeadEndsQuery, tree); err != nil {
		return nil, err
	}
	if d.Claims, err = treeClaims(conn, tree); err != nil {
		return nil, err
	}
	if len(d.Counts) == 0 && len(d.Claims) == 0 {
		return nil, nil
	}
	if len(d.Counts) == 0 {
		d.Counts = nil
	}
	return d, nil
}

func headlines(conn *sql.DB, q string, tree TreeID) ([]EntryHeadline, error) {
	rows, err := conn.Query(q, string(tree), digestLatestEntries)
	if err != nil {
		return nil, errs.WrapMessage("failed to read scratchpad headlines", err, "tree", string(tree))
	}
	defer rows.Close()
	var out []EntryHeadline
	for rows.Next() {
		var e EntryHeadline
		if err := rows.Scan(&e.ID, &e.Type, &e.Author, &e.Headline); err != nil {
			return nil, errs.WrapMessage("failed to read scratchpad headline", err, "tree", string(tree))
		}
		e.Headline = truncateBytes(strings.Join(strings.Fields(e.Headline), " "), headlineMaxBytes)
		out = append(out, e)
	}
	return out, rows.Err()
}

func treeClaims(conn *sql.DB, tree TreeID) ([]ClaimView, error) {
	rows, err := conn.Query(selectTreeClaimsQuery, string(tree), digestMaxClaims)
	if err != nil {
		return nil, errs.WrapMessage("failed to read handoff claims", err, "tree", string(tree))
	}
	var claims []ClaimView
	byKey := map[string]int{}
	for rows.Next() {
		var c ClaimView
		if rows.Scan(&c.Key, &c.Holder, &c.HolderLabel, &c.Reason, &c.GrantedAt) == nil {
			byKey[c.Key] = len(claims)
			claims = append(claims, c)
		}
	}
	rows.Close()
	if len(claims) == 0 {
		return nil, nil
	}
	rows, err = conn.Query(selectTreeClaimQueueQuery, string(tree))
	if err != nil {
		return nil, errs.WrapMessage("failed to read handoff claim queue", err, "tree", string(tree))
	}
	defer rows.Close()
	for rows.Next() {
		var key string
		var q QueuedClaim
		if rows.Scan(&key, &q.SessionID, &q.Reason, &q.EnqueuedAt) != nil {
			continue
		}
		if i, ok := byKey[key]; ok {
			q.Position = len(claims[i].Queue) + 1
			claims[i].Queue = append(claims[i].Queue, q)
		}
	}
	return claims, rows.Err()
}

// addDelivered adds tokens to sid's delivered total (OB-2), counts the read as an access to the
// tree (RQ-1), and returns sid's running totals.
func addDelivered(sid SessionID, tokens int) (available, delivered int, err error) {
	err = db.HandoffTx(func(tx *sql.Tx) error {
		if _, err := tx.Exec(addDeliveredQuery, tokens, string(sid)); err != nil {
			return errs.WrapMessage("failed to record handoff tokens delivered", err, "session_id", string(sid))
		}
		if _, err := tx.Exec(touchChildTreeQuery, sqlTime(nowFunc()), string(sid)); err != nil {
			return errs.WrapMessage("failed to touch handoff tree", err, "session_id", string(sid))
		}
		return tx.QueryRow(selectDeliveryQuery, string(sid)).Scan(&available, &delivered)
	})
	return available, delivered, err
}

// mappedProject is where a child's pointers resolve: its own project when that is a sibling
// worktree of the snapshot's (OP-8), else the snapshot's.
func mappedProject(snapshotProject, childProject string) string {
	if childProject == "" || snapshotProject == "" || childProject == snapshotProject {
		return snapshotProject
	}
	if repokey.SameRepo(childProject, snapshotProject) {
		return childProject
	}
	return snapshotProject
}

func responseTokens(v any) int {
	b, _ := json.Marshal(v)
	return db.EstimateTokens(string(b))
}
