package handoff

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// maxEntryTokens caps a posted entry's text (SP-2).
	maxEntryTokens = 500
	maxEntryRefs   = 20
	// maxReadEntries bounds one read page regardless of budget.
	maxReadEntries = 200
	// maxReadBudgetTokens clamps a caller's token_budget.
	maxReadBudgetTokens = 16000
	// maxDeadEnds bounds the dead-ends view (SP-7).
	maxDeadEnds = 10
	// entryOverheadTokens approximates the id, author, type, and time each entry carries besides
	// its text.
	entryOverheadTokens = 10

	insertEntryQuery = `INSERT INTO scratchpad_entries (tree_id, author_session_id, type, text, refs_json, token_est, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`
	entryColumns = `id, author_session_id, type, text, COALESCE(refs_json, ''), token_est, created_at, retracted_at IS NOT NULL`
	// The filters are bound, not spliced: an empty exclude_author or author disables its
	// filter, and types is a JSON array that is ignored when empty.
	selectEntriesQuery = `SELECT ` + entryColumns + ` FROM scratchpad_entries
		WHERE tree_id = ? AND id > ?
		AND (? OR retracted_at IS NULL)
		AND author_session_id <> ?
		AND (? = '' OR author_session_id = ?)
		AND (json_array_length(?) = 0 OR type IN (SELECT value FROM json_each(?)))
		ORDER BY id LIMIT ?`
	selectDeadEndEntriesQuery = `SELECT ` + entryColumns + ` FROM scratchpad_entries
		WHERE tree_id = ? AND retracted_at IS NULL AND author_session_id <> ?
		AND (type = '` + string(EntryTypeDeadEnd) + `'
			OR (type = '` + string(EntryTypeTrail) + `' AND json_extract(refs_json, '$.zero_hit') = 1))
		ORDER BY id DESC LIMIT ?`
	selectSnapshotTrailQuery = `SELECT h.parent_session_id, h.created_at, COALESCE(i.content, '')
		FROM handoff_snapshot_items i JOIN handoffs h ON h.ref = i.handoff_ref
		WHERE i.handoff_ref = ? AND i.section = '` + string(SectionTrail) + `' ORDER BY i.ord`
	selectEntryAuthorQuery = `SELECT author_session_id, retracted_at IS NOT NULL FROM scratchpad_entries WHERE id = ? AND tree_id = ?`
	retractEntryQuery      = `UPDATE scratchpad_entries SET retracted_at = ? WHERE id = ? AND retracted_at IS NULL`
)

// snapshotTrailJSON is a snapshot trail item's content (handoff_snapshot_items.content).
type snapshotTrailJSON struct {
	Tool     string   `json:"tool"`
	Query    string   `json:"query"`
	HitCount int      `json:"hit_count"`
	ZeroHit  bool     `json:"zero_hit"`
	TopHits  []string `json:"top_hits,omitempty"`
}

// Post appends a finding or dead end to the caller's tree scratchpad, charging the tree's caps
// in the same transaction (SP-1, SP-2, RQ-4). Claim and trail entries are written by the server.
func (s *realService) Post(ctx context.Context, req PostRequest) (*PostResponse, error) {
	te, ok := s.trees.lookup(req.SessionID)
	if !ok {
		return nil, errNotInTree(req.SessionID)
	}
	if req.Type != EntryTypeFinding && req.Type != EntryTypeDeadEnd {
		return nil, errs.NewCode(errs.CodeInvalidInput, "entry type must be finding or dead_end", "type", string(req.Type))
	}
	text := strings.TrimSpace(req.Text)
	if text == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "entry text is empty")
	}
	if n := db.EstimateTokens(text); n > maxEntryTokens {
		return nil, errs.NewCode(errs.CodeInvalidInput, "entry text is too long", "tokens", n, "max_tokens", maxEntryTokens)
	}
	refs := cleanRefs(req.Refs)
	if len(refs) > maxEntryRefs {
		return nil, errs.NewCode(errs.CodeInvalidInput, "too many refs", "refs", len(refs), "max_refs", maxEntryRefs)
	}
	refsJSON := ""
	if len(refs) > 0 {
		data, _ := json.Marshal(refs)
		refsJSON = string(data)
	}
	tokens := db.EstimateTokens(text) + db.EstimateTokens(strings.Join(refs, " "))
	res := &PostResponse{TreeID: te.tree, TokenEst: tokens}
	err := db.HandoffTx(func(tx *sql.Tx) error {
		if err := s.chargeTreeTx(tx, te.tree, tokens, 1); err != nil {
			return err
		}
		id, err := insertEntryTx(tx, te.tree, req.SessionID, req.Type, text, refsJSON, tokens)
		if err != nil {
			return err
		}
		u, err := treeUsageTx(tx, te.tree)
		if err != nil {
			return err
		}
		res.ID, res.TokensUsed, res.EntriesUsed = id, u.tokens, u.entries
		return nil
	})
	if err != nil {
		return nil, err
	}
	s.logger.Debug("Posted scratchpad entry", req.SessionID.Attr(), te.tree.Attr(), "entry", res.ID, "type", string(req.Type))
	return res, nil
}

// Read returns the tree's entries after the Since cursor within the token budget, oldest first
// (SP-4). The caller's own entries are excluded unless IncludeOwn is set or Author names the
// caller. When Types asks for dead ends, DeadEnds carries the dead-ends view (SP-7); active
// claims and their queues are always included (CL-8).
func (s *realService) Read(ctx context.Context, req ReadRequest) (*ReadResponse, error) {
	te, ok := s.trees.lookup(req.SessionID)
	if !ok {
		return nil, errNotInTree(req.SessionID)
	}
	if req.Since < 0 {
		return nil, errs.NewCode(errs.CodeInvalidInput, "since must not be negative", "since", req.Since)
	}
	for _, t := range req.Types {
		if !t.Valid() {
			return nil, errs.NewCode(errs.CodeInvalidInput, "unknown entry type", "type", string(t))
		}
	}
	conn := db.ContextDB
	if conn == nil {
		return nil, errNoContextDB
	}
	budget := readBudget(req.TokenBudget)
	exclude := string(req.SessionID)
	if req.IncludeOwn || req.Author == req.SessionID {
		exclude = ""
	}
	typesJSON, _ := json.Marshal(nonNil(req.Types))
	rows, err := conn.Query(selectEntriesQuery, string(te.tree), req.Since, req.IncludeRetracted, exclude,
		string(req.Author), string(req.Author), string(typesJSON), string(typesJSON), maxReadEntries+1)
	if err != nil {
		return nil, errs.WrapMessage("failed to read scratchpad", err, "tree", string(te.tree))
	}
	page, err := scanEntries(rows)
	if err != nil {
		return nil, errs.WrapMessage("failed to read scratchpad entry", err, "tree", string(te.tree))
	}
	res := &ReadResponse{Entries: []ScratchpadEntry{}, NextCursor: req.Since}
	for i, e := range page {
		cost := e.TokenEst + entryOverheadTokens
		// The first entry always goes out, so a small budget still makes progress.
		if i == maxReadEntries || (len(res.Entries) > 0 && res.TokensUsed+cost > budget) {
			res.Truncated = true
			break
		}
		res.Entries = append(res.Entries, e)
		res.TokensUsed += cost
		res.NextCursor = e.ID
	}
	if slices.Contains(req.Types, EntryTypeDeadEnd) {
		views, err := s.deadEnds(conn, te, exclude)
		if err != nil {
			return nil, err
		}
		res.DeadEnds = fitDeadEnds(views, res.Entries, budget, &res.TokensUsed)
	}
	if res.Claims, err = claimViews(conn, te.tree); err != nil {
		return nil, err
	}
	if len(res.Claims) > 0 {
		data, _ := json.Marshal(res.Claims)
		res.TokensUsed += db.EstimateTokens(string(data))
	}
	return res, nil
}

// Retract marks one of the caller's own entries retracted; it stays stored but reads hide it
// (SP-3). Retracting an already retracted entry succeeds.
func (s *realService) Retract(ctx context.Context, req RetractRequest) (*RetractResponse, error) {
	te, ok := s.trees.lookup(req.SessionID)
	if !ok {
		return nil, errNotInTree(req.SessionID)
	}
	err := db.HandoffTx(func(tx *sql.Tx) error {
		var author SessionID
		var retracted bool
		err := tx.QueryRow(selectEntryAuthorQuery, req.Entry, string(te.tree)).Scan(&author, &retracted)
		if errors.Is(err, sql.ErrNoRows) {
			return errs.NewCode(errs.CodeNotFound, "scratchpad entry not found", "entry", req.Entry, "tree", string(te.tree))
		}
		if err != nil {
			return errs.WrapMessage("failed to read scratchpad entry", err, "entry", req.Entry)
		}
		if author != req.SessionID {
			return errs.NewCode(errs.CodeInvalidInput, "only the author can retract an entry", "entry", req.Entry, "author", string(author))
		}
		if retracted {
			return nil
		}
		if _, err := tx.Exec(retractEntryQuery, sqlTime(nowFunc()), req.Entry); err != nil {
			return errs.WrapMessage("failed to retract scratchpad entry", err, "entry", req.Entry)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &RetractResponse{Entry: req.Entry, Retracted: true}, nil
}

// deadEnds builds the dead-ends view (SP-7): the tree's newest dead_end posts and zero-hit trail
// entries, then, for a child, the zero-hit searches in the snapshot it opened. Entries by
// exclude are left out.
func (s *realService) deadEnds(conn *sql.DB, te treeEntry, exclude string) ([]ScratchpadEntry, error) {
	rows, err := conn.Query(selectDeadEndEntriesQuery, string(te.tree), exclude, maxDeadEnds+maxReadEntries)
	if err != nil {
		return nil, errs.WrapMessage("failed to read scratchpad dead ends", err, "tree", string(te.tree))
	}
	out, err := scanEntries(rows)
	if err != nil {
		return nil, errs.WrapMessage("failed to read scratchpad dead end", err, "tree", string(te.tree))
	}
	if !te.isChild {
		return out, nil
	}
	snap, err := snapshotDeadEnds(conn, te.handoff, exclude)
	if err != nil {
		return nil, err
	}
	return append(out, snap...), nil
}

// snapshotDeadEnds returns the zero-hit trail items of ref's snapshot as entries authored by the
// parent. They have no scratchpad id, so ID is 0.
func snapshotDeadEnds(conn *sql.DB, ref HandoffRef, exclude string) ([]ScratchpadEntry, error) {
	rows, err := conn.Query(selectSnapshotTrailQuery, string(ref))
	if err != nil {
		return nil, errs.WrapMessage("failed to read snapshot trail", err, "handoff", string(ref))
	}
	defer rows.Close()
	var out []ScratchpadEntry
	for rows.Next() {
		var parent SessionID
		var createdAt, content string
		if err := rows.Scan(&parent, &createdAt, &content); err != nil {
			return nil, errs.WrapMessage("failed to read snapshot trail item", err, "handoff", string(ref))
		}
		var item snapshotTrailJSON
		if json.Unmarshal([]byte(content), &item) != nil || !item.ZeroHit || string(parent) == exclude {
			continue
		}
		text := trailEntryText(item.Tool, item.Query, item.HitCount)
		out = append(out, ScratchpadEntry{
			Author: parent, Type: EntryTypeTrail, Text: text, Refs: item.TopHits,
			TokenEst: db.EstimateTokens(text), CreatedAt: createdAt,
		})
	}
	return out, rows.Err()
}

// fitDeadEnds keeps up to maxDeadEnds views not already in page, within what is left of budget,
// adding their cost to used.
func fitDeadEnds(views, page []ScratchpadEntry, budget int, used *int) []ScratchpadEntry {
	inPage := make(map[int64]bool, len(page))
	for _, e := range page {
		inPage[e.ID] = true
	}
	var out []ScratchpadEntry
	for _, e := range views {
		if e.ID != 0 && inPage[e.ID] {
			continue
		}
		cost := e.TokenEst + entryOverheadTokens
		if len(out) == maxDeadEnds || *used+cost > budget {
			break
		}
		out = append(out, e)
		*used += cost
	}
	return out
}

func insertEntryTx(tx *sql.Tx, tree TreeID, author SessionID, typ EntryType, text, refsJSON string, tokens int) (int64, error) {
	var refs any
	if refsJSON != "" {
		refs = refsJSON
	}
	r, err := tx.Exec(insertEntryQuery, string(tree), string(author), string(typ), text, refs, tokens, sqlTime(nowFunc()))
	if err != nil {
		return 0, errs.WrapMessage("failed to insert scratchpad entry", err, "tree", string(tree), "type", string(typ))
	}
	id, err := r.LastInsertId()
	if err != nil {
		return 0, errs.WrapMessage("failed to read scratchpad entry id", err, "tree", string(tree))
	}
	return id, nil
}

func scanEntries(rows *sql.Rows) ([]ScratchpadEntry, error) {
	defer rows.Close()
	var out []ScratchpadEntry
	for rows.Next() {
		var e ScratchpadEntry
		var refs string
		if err := rows.Scan(&e.ID, &e.Author, &e.Type, &e.Text, &refs, &e.TokenEst, &e.CreatedAt, &e.Retracted); err != nil {
			return nil, err
		}
		e.Refs = decodeRefs(refs)
		out = append(out, e)
	}
	return out, rows.Err()
}

// decodeRefs reads refs_json: a JSON array of refs for posts and claims, or a trail entry's
// object, whose top hits serve as its refs.
func decodeRefs(raw string) []string {
	if raw == "" {
		return nil
	}
	var refs []string
	if json.Unmarshal([]byte(raw), &refs) == nil {
		return refs
	}
	var t liveTrailRefs
	if json.Unmarshal([]byte(raw), &t) == nil {
		return t.TopHits
	}
	return nil
}

func cleanRefs(refs []string) []string {
	out := make([]string, 0, len(refs))
	for _, r := range refs {
		if r = strings.TrimSpace(r); r != "" && !slices.Contains(out, r) {
			out = append(out, r)
		}
	}
	return out
}

func readBudget(requested int) int {
	if requested <= 0 {
		return LoadLimits().OpenBudgetTokens
	}
	return min(requested, maxReadBudgetTokens)
}

// trailEntryText is a trail entry's one-line text: "<tool>: <query> (<n> hits)".
func trailEntryText(tool, query string, hits int) string {
	return tool + ": " + query + " (" + strconv.Itoa(hits) + " hits)"
}

func nonNil[T any](s []T) []T {
	if s == nil {
		return []T{}
	}
	return s
}

func errNotInTree(sid SessionID) error {
	return errs.NewCode(CodeHandoffNotFound, "session is not in a handoff tree", "session", string(sid))
}
