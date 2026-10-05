package handoff

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"unicode/utf8"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

const (
	// stubLabelMaxBytes keeps the stub under NFR-3's 60 tokens whatever the label.
	stubLabelMaxBytes = 120
	// derivedLabelWords is how much of the brief's first line becomes a missing label.
	derivedLabelWords = 8

	selectParentChildQuery = `SELECT c.tree_id, h.depth, t.last_access_at
		FROM handoff_children c JOIN handoffs h ON h.ref = c.handoff_ref JOIN handoff_trees t ON t.tree_id = c.tree_id
		WHERE c.child_session_id = ?`
	selectChildProjectQuery   = `SELECT COALESCE(project_path, '') FROM handoff_children WHERE child_session_id = ?`
	selectRootTreeAccessQuery = `SELECT tree_id, last_access_at FROM handoff_trees WHERE root_session_id = ?
		ORDER BY created_at DESC LIMIT 1`
	insertTreeQuery = `INSERT INTO handoff_trees (tree_id, root_session_id, project_path, created_at, last_access_at)
		VALUES (?, ?, ?, ?, ?)`
	insertHandoffQuery = `INSERT INTO handoffs (ref, tree_id, parent_session_id, parent_child_session_id, depth, mode, label,
		brief, project_path, child_count, created_at, last_access_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`
	insertSnapshotItemQuery = `INSERT INTO handoff_snapshot_items (handoff_ref, section, ord, item_key, label, content, file_rel,
		fqn, kind, start_line, end_line, fingerprint, token_est) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
)

// placement is where a new handoff sits: its tree, depth, and the child session that created
// it when it is nested (HO-8).
type placement struct {
	tree        TreeID
	depth       int
	parentChild SessionID
}

// Create snapshots the parent session and stores the handoff (HO-1–HO-10). The snapshot is read
// first, then the tree, handoff, items, and tree usage are written in one transaction, so a
// failure stores nothing (HO-6).
func (s *realService) Create(ctx context.Context, req CreateRequest) (*CreateResponse, error) {
	if err := validateCreate(&req); err != nil {
		return nil, err
	}
	project, err := createProject(req)
	if err != nil {
		return nil, err
	}
	snap, err := gatherSnapshot(req, project)
	if err != nil {
		return nil, err
	}
	l := LoadLimits()
	if snap.breakdown.Total > l.TreeMaxTokens || snap.entries > l.TreeMaxEntries {
		return nil, errs.NewCode(CodeHandoffTreeLimitExceeded, "handoff snapshot exceeds the tree cap",
			"breakdown", snap.breakdown, "entries", snap.entries, "tokens_max", l.TreeMaxTokens, "entries_max", l.TreeMaxEntries)
	}
	ref, err := NewHandoffRef()
	if err != nil {
		return nil, err
	}
	label := req.Label
	if label == "" {
		label = deriveLabel(req.Brief)
	}
	var place placement
	var usedTokens int
	err = db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		if place, err = placeHandoffTx(tx, req.SessionID, project, l); err != nil {
			return err
		}
		if usedTokens, err = s.chargeTreeTokensTx(tx, place.tree, snap.breakdown.Total, snap.entries); err != nil {
			return errs.Wrap(err, "breakdown", snap.breakdown)
		}
		return insertHandoffTx(tx, ref, place, req, label, project, snap)
	})
	if err != nil {
		return nil, err
	}
	if place.parentChild == "" {
		s.trees.put(req.SessionID, treeEntry{tree: place.tree})
	}
	handoffsCreated.Inc()
	treeTokens.Observe(float64(usedTokens))
	notifyDashboard()
	s.logger.Info("Created handoff", lifecycleArgs(place.tree, ref, req.SessionID, "", project,
		"depth", place.depth, "mode", string(req.Mode), "tokens", snap.breakdown.Total, "entries", snap.entries)...)
	return &CreateResponse{Ref: ref, TreeID: place.tree, Depth: place.depth, Stub: handoffStub(ref, label), Breakdown: snap.breakdown}, nil
}

func validateCreate(req *CreateRequest) error {
	req.SessionID = SessionID(strings.TrimSpace(string(req.SessionID)))
	req.Brief = strings.TrimSpace(req.Brief)
	req.Label = strings.TrimSpace(req.Label)
	if req.SessionID == "" {
		return errs.NewCode(errs.CodeInvalidInput, "session_id required")
	}
	if req.Brief == "" {
		return errs.NewCode(errs.CodeInvalidInput, "brief required")
	}
	if req.Mode == "" {
		req.Mode = ModeFresh
	}
	if !req.Mode.Valid() {
		return errs.NewCode(errs.CodeInvalidInput, "mode must be fresh or fork", "mode", string(req.Mode))
	}
	return nil
}

// createProject is the request's project, or, for a child creating a nested handoff without
// one, the project it opened its own handoff in.
func createProject(req CreateRequest) (string, error) {
	if p := projectlinks.NormalizePath(req.ProjectPath); p != "" {
		return p, nil
	}
	if db.ContextDB == nil {
		return "", errNoContextDB
	}
	var p string
	err := db.ContextDB.QueryRow(selectChildProjectQuery, string(req.SessionID)).Scan(&p)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return "", errs.WrapMessage("failed to read child session project", err, "session_id", string(req.SessionID))
	}
	return p, nil
}

// placeHandoffTx finds the tree for a handoff created by sid. A child nests in its own tree one
// level deeper, up to handoff_max_depth (HO-8); a root reuses its live tree (one tree per root)
// or starts one.
func placeHandoffTx(tx *sql.Tx, sid SessionID, project string, l Limits) (placement, error) {
	now := nowFunc()
	cutoff := sqlTime(now.Add(-l.TTL()))
	var p placement
	var parentDepth int
	var access string
	err := tx.QueryRow(selectParentChildQuery, string(sid)).Scan(&p.tree, &parentDepth, &access)
	if err == nil {
		if access < cutoff {
			return p, errs.NewCode(CodeHandoffExpired, "handoff tree expired", "tree", string(p.tree))
		}
		p.depth, p.parentChild = parentDepth+1, sid
		if p.depth > l.MaxDepth {
			return p, errs.NewCode(CodeHandoffDepthExceeded, "handoff depth exceeded", "session_id", string(sid),
				"depth", p.depth, "max_depth", l.MaxDepth)
		}
		return p, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return p, errs.WrapMessage("failed to read parent child session", err, "session_id", string(sid))
	}
	p.depth = 1
	err = tx.QueryRow(selectRootTreeAccessQuery, string(sid)).Scan(&p.tree, &access)
	if err == nil && access >= cutoff {
		return p, nil
	}
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return p, errs.WrapMessage("failed to read root session tree", err, "session_id", string(sid))
	}
	// No tree yet, or only an expired one the sweeper hasn't flushed: start a new one.
	if p.tree, err = NewTreeID(); err != nil {
		return p, err
	}
	ts := sqlTime(now)
	if _, err := tx.Exec(insertTreeQuery, string(p.tree), string(sid), nullIfEmpty(project), ts, ts); err != nil {
		return p, errs.WrapMessage("failed to create handoff tree", err, "session_id", string(sid))
	}
	return p, nil
}

func insertHandoffTx(tx *sql.Tx, ref HandoffRef, p placement, req CreateRequest, label, project string, snap *snapshot) error {
	ts := sqlTime(nowFunc())
	if _, err := tx.Exec(insertHandoffQuery, string(ref), string(p.tree), string(req.SessionID), nullIfEmpty(string(p.parentChild)),
		p.depth, string(req.Mode), nullIfEmpty(label), req.Brief, nullIfEmpty(project), ts, ts); err != nil {
		return errs.WrapMessage("failed to insert handoff", err, "handoff", string(ref))
	}
	stmt, err := tx.Prepare(insertSnapshotItemQuery)
	if err != nil {
		return errs.WrapMessage("failed to prepare snapshot insert", err, "handoff", string(ref))
	}
	defer stmt.Close()
	for _, it := range snap.items {
		if _, err := stmt.Exec(string(ref), string(it.section), it.ord, it.key, nullIfEmpty(it.label), nullIfEmpty(it.content),
			nullIfEmpty(it.fileRel), nullIfEmpty(it.fqn), nullIfEmpty(it.kind), it.startLine, it.endLine,
			nullIfEmpty(it.fingerprint), it.tokenEst); err != nil {
			return errs.WrapMessage("failed to insert snapshot item", err, "handoff", string(ref), "section", string(it.section))
		}
	}
	if _, err := tx.Exec(touchTreeQuery, ts, string(p.tree)); err != nil {
		return errs.WrapMessage("failed to touch handoff tree", err, "tree", string(p.tree))
	}
	return nil
}

// handoffStub is the HO-5 prompt stub; the label is cut so the stub stays within 60 tokens.
func handoffStub(ref HandoffRef, label string) string {
	return "[handoff " + string(ref) + "] " + truncateBytes(label, stubLabelMaxBytes) + " — call open_handoff first"
}

// deriveLabel is the first few words of the brief's first line.
func deriveLabel(brief string) string {
	line, _, _ := strings.Cut(brief, "\n")
	words := strings.Fields(line)
	if len(words) > derivedLabelWords {
		words = append(words[:derivedLabelWords], "…")
	}
	return strings.Join(words, " ")
}

// truncateBytes cuts s to at most n bytes on a rune boundary, marking the cut with "…".
func truncateBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= len("…") {
		return ""
	}
	cut := n - len("…")
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}
