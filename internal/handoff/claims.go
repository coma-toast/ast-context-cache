package handoff

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	maxClaimKeyBytes    = 512
	maxClaimReasonBytes = 200

	selectClaimHolderQuery = `SELECT holder_session_id FROM handoff_claims WHERE tree_id = ? AND key = ?`
	insertClaimQuery       = `INSERT INTO handoff_claims (tree_id, key, holder_session_id, reason, granted_at) VALUES (?, ?, ?, ?, ?)`
	deleteClaimQuery       = `DELETE FROM handoff_claims WHERE tree_id = ? AND key = ? AND holder_session_id = ?`
	selectHeldKeysQuery    = `SELECT key FROM handoff_claims WHERE tree_id = ? AND holder_session_id = ? ORDER BY key`
	deleteHeldClaimsQuery  = `DELETE FROM handoff_claims WHERE tree_id = ? AND holder_session_id = ?`
	selectChildLabelQuery  = `SELECT COALESCE(label, '') FROM handoff_children WHERE child_session_id = ?`

	selectQueuedIDQuery      = `SELECT id FROM handoff_claim_queue WHERE tree_id = ? AND key = ? AND session_id = ? ORDER BY id LIMIT 1`
	selectQueuePositionQuery = `SELECT COUNT(*) FROM handoff_claim_queue WHERE tree_id = ? AND key = ? AND id <= ?`
	insertQueueQuery         = `INSERT INTO handoff_claim_queue (tree_id, key, session_id, reason, enqueued_at) VALUES (?, ?, ?, ?, ?)`
	selectQueueHeadQuery     = `SELECT id, session_id, COALESCE(reason, ''), enqueued_at FROM handoff_claim_queue
		WHERE tree_id = ? AND key = ? ORDER BY id LIMIT 1`
	deleteQueueRowQuery     = `DELETE FROM handoff_claim_queue WHERE id = ?`
	deleteQueuedKeyQuery    = `DELETE FROM handoff_claim_queue WHERE tree_id = ? AND key = ? AND session_id = ?`
	deleteSessionQueueQuery = `DELETE FROM handoff_claim_queue WHERE tree_id = ? AND session_id = ?`
	// selectWaitEdgesQuery lists the tree's wait-for edges: each waiter and the holder of the
	// key it waits for.
	selectWaitEdgesQuery = `SELECT q.session_id, q.key, c.holder_session_id FROM handoff_claim_queue q
		JOIN handoff_claims c ON c.tree_id = q.tree_id AND c.key = q.key
		WHERE q.tree_id = ? ORDER BY q.id`

	insertGrantQuery         = `INSERT INTO handoff_claim_grants (session_id, tree_id, key, granted_at) VALUES (?, ?, ?, ?)`
	hasPendingGrantsQuery    = `SELECT EXISTS(SELECT 1 FROM handoff_claim_grants WHERE session_id = ? AND notified_at IS NULL)`
	selectPendingGrantsQuery = `SELECT id, tree_id, key, granted_at FROM handoff_claim_grants
		WHERE session_id = ? AND notified_at IS NULL ORDER BY id`
	markGrantNotifiedQuery = `UPDATE handoff_claim_grants SET notified_at = ? WHERE id = ?`

	selectClaimViewsQuery = `SELECT c.key, c.holder_session_id, COALESCE(ch.label, ''), COALESCE(c.reason, ''), c.granted_at
		FROM handoff_claims c LEFT JOIN handoff_children ch ON ch.child_session_id = c.holder_session_id
		WHERE c.tree_id = ? ORDER BY c.key`
	selectClaimQueuesQuery = `SELECT key, session_id, COALESCE(reason, ''), enqueued_at FROM handoff_claim_queue
		WHERE tree_id = ? ORDER BY key, id`
)

// onClaimWait observes how long each queued claim waited before it was granted: the
// claim_wait_seconds histogram, which tests swap out; nil means unobserved.
var onClaimWait = observeClaimWaitSeconds

// waitEdge is one wait-for edge: the waiter waits for key, held by to.
type waitEdge struct {
	to  SessionID
	key string
}

// querier is satisfied by both *sql.DB and *sql.Tx.
type querier interface {
	Query(query string, args ...any) (*sql.Rows, error)
}

// Claim takes an advisory claim on a key in the caller's tree, or queues for it FIFO behind its
// holder (CL-1, CL-2). Claiming a key the caller holds or already waits for reports its current
// state. A request that would close a wait cycle is rejected with CodeClaimDeadlockRisk and is
// not queued (CL-6).
func (s *realService) Claim(ctx context.Context, req ClaimRequest) (*ClaimResponse, error) {
	te, ok := s.trees.lookup(req.SessionID)
	if !ok {
		return nil, errNotInTree(req.SessionID)
	}
	key, err := normalizeClaimKey(req.Key)
	if err != nil {
		return nil, err
	}
	reason := cutBytes(strings.TrimSpace(req.Reason), maxClaimReasonBytes)
	var res *ClaimResponse
	err = db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		res, err = s.claimTx(tx, te.tree, req.SessionID, key, reason)
		return err
	})
	if err != nil {
		return nil, err
	}
	if res.Outcome == ClaimHeld {
		s.logger.Debug("Handled claim", sessionEventArgs(contextReader(), req.SessionID, te, "key", key, "outcome", string(res.Outcome))...)
		return res, nil
	}
	notifyDashboard()
	if res.Outcome == ClaimQueued {
		s.logger.Debug("Queued claim", sessionEventArgs(contextReader(), req.SessionID, te, "key", key, "holder", string(res.Holder),
			"position", res.Position)...)
		return res, nil
	}
	s.logger.Info("Granted claim", sessionEventArgs(contextReader(), req.SessionID, te, "key", key)...)
	return res, nil
}

// Release gives up the caller's claim on a key, granting it to the next waiter (CL-3, CL-4), or
// withdraws the caller from the key's queue.
func (s *realService) Release(ctx context.Context, req ReleaseRequest) (*ReleaseResponse, error) {
	te, ok := s.trees.lookup(req.SessionID)
	if !ok {
		return nil, errNotInTree(req.SessionID)
	}
	key, err := normalizeClaimKey(req.Key)
	if err != nil {
		return nil, err
	}
	res := &ReleaseResponse{Key: key}
	err = db.HandoffTx(func(tx *sql.Tx) error {
		res.Released, res.GrantedTo = false, ""
		r, err := tx.Exec(deleteClaimQuery, string(te.tree), key, string(req.SessionID))
		if err != nil {
			return errs.WrapMessage("failed to release claim", err, "key", key)
		}
		if n, _ := r.RowsAffected(); n > 0 {
			res.Released = true
			res.GrantedTo, err = s.grantNextTx(tx, te.tree, key, req.SessionID)
			return err
		}
		r, err = tx.Exec(deleteQueuedKeyQuery, string(te.tree), key, string(req.SessionID))
		if err != nil {
			return errs.WrapMessage("failed to leave claim queue", err, "key", key)
		}
		if n, _ := r.RowsAffected(); n == 0 {
			return errs.NewCode(errs.CodeNotFound, "no claim or queued request for key", "key", key, "session", string(req.SessionID))
		}
		res.Released = true
		return nil
	})
	if err != nil {
		return nil, err
	}
	notifyDashboard()
	s.logger.Info("Released claim", sessionEventArgs(contextReader(), req.SessionID, te, "key", key, "granted_to", string(res.GrantedTo))...)
	return res, nil
}

// PendingGrants returns the claims granted to sid from a queue since it was last told, and
// marks them told (CL-4, CL-5). Sessions outside a tree, and the common case of nothing
// pending, cost no write transaction.
func (s *realService) PendingGrants(sid SessionID) ([]Grant, error) {
	if !s.IsTreeSession(sid) {
		return nil, nil
	}
	conn := db.ContextDB
	if conn == nil {
		return nil, errNoContextDB
	}
	var pending bool
	if err := conn.QueryRow(hasPendingGrantsQuery, string(sid)).Scan(&pending); err != nil {
		return nil, errs.WrapMessage("failed to check pending claim grants", err, "session", string(sid))
	}
	if !pending {
		return nil, nil
	}
	var out []Grant
	err := db.HandoffTx(func(tx *sql.Tx) error {
		out = nil
		rows, err := tx.Query(selectPendingGrantsQuery, string(sid))
		if err != nil {
			return errs.WrapMessage("failed to read pending claim grants", err, "session", string(sid))
		}
		var ids []int64
		for rows.Next() {
			var id int64
			var g Grant
			if err := rows.Scan(&id, &g.TreeID, &g.Key, &g.GrantedAt); err != nil {
				rows.Close()
				return errs.WrapMessage("failed to read pending claim grant", err, "session", string(sid))
			}
			ids, out = append(ids, id), append(out, g)
		}
		rows.Close()
		now := sqlTime(nowFunc())
		for _, id := range ids {
			if _, err := tx.Exec(markGrantNotifiedQuery, now, id); err != nil {
				return errs.WrapMessage("failed to mark claim grant notified", err, "session", string(sid))
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// claimTx grants, reports, or queues sid's claim on key.
func (s *realService) claimTx(tx *sql.Tx, tree TreeID, sid SessionID, key, reason string) (*ClaimResponse, error) {
	res := &ClaimResponse{Key: key}
	holder, err := claimHolderTx(tx, tree, key)
	if err != nil {
		return nil, err
	}
	switch holder {
	case "":
		if _, err := tx.Exec(insertClaimQuery, string(tree), key, string(sid), nullable(reason), sqlTime(nowFunc())); err != nil {
			return nil, errs.WrapMessage("failed to insert claim", err, "key", key)
		}
		res.Outcome, res.Holder = ClaimGranted, sid
		return res, s.claimEntryTx(tx, tree, sid, key, "claimed "+key)
	case sid:
		res.Outcome, res.Holder = ClaimHeld, sid
		return res, nil
	}
	res.Outcome, res.Holder = ClaimQueued, holder
	if res.HolderLabel, err = childLabelTx(tx, holder); err != nil {
		return nil, err
	}
	var qid int64
	err = tx.QueryRow(selectQueuedIDQuery, string(tree), key, string(sid)).Scan(&qid)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, errs.WrapMessage("failed to read claim queue", err, "key", key)
	}
	if qid == 0 {
		if qid, err = s.enqueueTx(tx, tree, sid, holder, key, reason); err != nil {
			return nil, err
		}
	}
	if err := tx.QueryRow(selectQueuePositionQuery, string(tree), key, qid).Scan(&res.Position); err != nil {
		return nil, errs.WrapMessage("failed to read claim queue position", err, "key", key)
	}
	return res, nil
}

// enqueueTx queues sid for key behind holder unless that would close a wait cycle (CL-6).
func (s *realService) enqueueTx(tx *sql.Tx, tree TreeID, sid, holder SessionID, key, reason string) (int64, error) {
	edges, err := waitEdgesTx(tx, tree)
	if err != nil {
		return 0, err
	}
	if path := waitPath(edges, holder, sid); path != nil {
		cycle := append([]string{describeWait(sid, key, holder)}, path...)
		return 0, errs.NewCode(CodeClaimDeadlockRisk, "claim would create a wait cycle", "key", key,
			"holder", string(holder), "cycle", strings.Join(cycle, "; "))
	}
	r, err := tx.Exec(insertQueueQuery, string(tree), key, string(sid), nullable(reason), sqlTime(nowFunc()))
	if err != nil {
		return 0, errs.WrapMessage("failed to queue claim", err, "key", key)
	}
	qid, err := r.LastInsertId()
	if err != nil {
		return 0, errs.WrapMessage("failed to read claim queue id", err, "key", key)
	}
	return qid, s.claimEntryTx(tx, tree, sid, key, "queued for "+key+" behind "+string(holder))
}

// releaseAllTx releases every claim sid holds in tree and leaves its queue positions, granting
// each released key to its next waiter, and returns the released keys (CL-3). Completion and
// abandonment call it inside their own transaction.
func (s *realService) releaseAllTx(tx *sql.Tx, tree TreeID, sid SessionID) ([]string, error) {
	rows, err := tx.Query(selectHeldKeysQuery, string(tree), string(sid))
	if err != nil {
		return nil, errs.WrapMessage("failed to list held claims", err, "session", string(sid))
	}
	var keys []string
	for rows.Next() {
		var k string
		if err := rows.Scan(&k); err != nil {
			rows.Close()
			return nil, errs.WrapMessage("failed to read held claim", err, "session", string(sid))
		}
		keys = append(keys, k)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, errs.WrapMessage("failed to list held claims", err, "session", string(sid))
	}
	if _, err := tx.Exec(deleteHeldClaimsQuery, string(tree), string(sid)); err != nil {
		return nil, errs.WrapMessage("failed to release claims", err, "session", string(sid))
	}
	if _, err := tx.Exec(deleteSessionQueueQuery, string(tree), string(sid)); err != nil {
		return nil, errs.WrapMessage("failed to leave claim queues", err, "session", string(sid))
	}
	for _, k := range keys {
		if _, err := s.grantNextTx(tx, tree, k, sid); err != nil {
			return nil, err
		}
	}
	return keys, nil
}

// grantNextTx pops key's queue head, grants it the claim, and records a grant for it to be told
// of (CL-4). It returns the grantee, or "" when nobody was waiting.
func (s *realService) grantNextTx(tx *sql.Tx, tree TreeID, key string, prev SessionID) (SessionID, error) {
	var id int64
	var next SessionID
	var reason, enqueuedAt string
	err := tx.QueryRow(selectQueueHeadQuery, string(tree), key).Scan(&id, &next, &reason, &enqueuedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", errs.WrapMessage("failed to read claim queue head", err, "key", key)
	}
	now := nowFunc()
	if _, err := tx.Exec(deleteQueueRowQuery, id); err != nil {
		return "", errs.WrapMessage("failed to pop claim queue", err, "key", key)
	}
	if _, err := tx.Exec(insertClaimQuery, string(tree), key, string(next), nullable(reason), sqlTime(now)); err != nil {
		return "", errs.WrapMessage("failed to grant queued claim", err, "key", key, "session", string(next))
	}
	if _, err := tx.Exec(insertGrantQuery, string(next), string(tree), key, sqlTime(now)); err != nil {
		return "", errs.WrapMessage("failed to record claim grant", err, "key", key, "session", string(next))
	}
	observeClaimWait(now, enqueuedAt)
	// Logged before the enclosing transaction commits: it is rolled back only when the rest of a
	// release, completion, or abandonment fails, which that caller logs.
	te, ok := s.trees.lookup(next)
	if !ok {
		te = treeEntry{tree: tree, isChild: true}
	}
	s.logger.Info("Granted queued claim", sessionEventArgs(tx, next, te, "key", key, "after", string(prev))...)
	return next, s.claimEntryTx(tx, tree, next, key, "granted "+key+" after "+string(prev)+" released it")
}

// claimEntryTx writes a claim entry so the tree sees claims in its scratchpad (CL-8). Claim
// entries are never evicted, but one is dropped rather than failing the claim when the tree has
// no room.
func (s *realService) claimEntryTx(tx *sql.Tx, tree TreeID, sid SessionID, key, text string) error {
	refs, _ := json.Marshal([]string{key})
	_, err := s.insertAutoEntryTx(tx, tree, sid, EntryTypeClaim, text, string(refs), db.EstimateTokens(text))
	return err
}

// claimViews lists tree's active claims with their queues, by key (CL-8).
func claimViews(q querier, tree TreeID) ([]ClaimView, error) {
	rows, err := q.Query(selectClaimViewsQuery, string(tree))
	if err != nil {
		return nil, errs.WrapMessage("failed to list claims", err, "tree", string(tree))
	}
	var out []ClaimView
	index := map[string]int{}
	for rows.Next() {
		var v ClaimView
		if err := rows.Scan(&v.Key, &v.Holder, &v.HolderLabel, &v.Reason, &v.GrantedAt); err != nil {
			rows.Close()
			return nil, errs.WrapMessage("failed to read claim", err, "tree", string(tree))
		}
		index[v.Key] = len(out)
		out = append(out, v)
	}
	rows.Close()
	if len(out) == 0 {
		return nil, rows.Err()
	}
	rows, err = q.Query(selectClaimQueuesQuery, string(tree))
	if err != nil {
		return nil, errs.WrapMessage("failed to list claim queues", err, "tree", string(tree))
	}
	defer rows.Close()
	for rows.Next() {
		var key string
		var w QueuedClaim
		if err := rows.Scan(&key, &w.SessionID, &w.Reason, &w.EnqueuedAt); err != nil {
			return nil, errs.WrapMessage("failed to read claim queue entry", err, "tree", string(tree))
		}
		i, ok := index[key]
		if !ok {
			continue
		}
		w.Position = len(out[i].Queue) + 1
		out[i].Queue = append(out[i].Queue, w)
	}
	return out, rows.Err()
}

func claimHolderTx(tx *sql.Tx, tree TreeID, key string) (SessionID, error) {
	var holder SessionID
	err := tx.QueryRow(selectClaimHolderQuery, string(tree), key).Scan(&holder)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", errs.WrapMessage("failed to read claim holder", err, "key", key)
	}
	return holder, nil
}

// childLabelTx returns a child's handoff label; a root session has none.
func childLabelTx(tx *sql.Tx, sid SessionID) (string, error) {
	var label string
	err := tx.QueryRow(selectChildLabelQuery, string(sid)).Scan(&label)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", errs.WrapMessage("failed to read child label", err, "session", string(sid))
	}
	return label, nil
}

func waitEdgesTx(tx *sql.Tx, tree TreeID) (map[SessionID][]waitEdge, error) {
	rows, err := tx.Query(selectWaitEdgesQuery, string(tree))
	if err != nil {
		return nil, errs.WrapMessage("failed to read claim wait graph", err, "tree", string(tree))
	}
	defer rows.Close()
	edges := map[SessionID][]waitEdge{}
	for rows.Next() {
		var from SessionID
		var e waitEdge
		if err := rows.Scan(&from, &e.key, &e.to); err != nil {
			return nil, errs.WrapMessage("failed to read claim wait edge", err, "tree", string(tree))
		}
		edges[from] = append(edges[from], e)
	}
	return edges, rows.Err()
}

// waitPath returns the waits leading from from to target in the wait-for graph, described one
// per step, or nil when target is unreachable. A depth-first search suffices: the graph has one
// node per tree session.
func waitPath(edges map[SessionID][]waitEdge, from, target SessionID) []string {
	seen := map[SessionID]bool{}
	var walk func(at SessionID) []string
	walk = func(at SessionID) []string {
		if seen[at] {
			return nil
		}
		seen[at] = true
		for _, e := range edges[at] {
			step := describeWait(at, e.key, e.to)
			if e.to == target {
				return []string{step}
			}
			if rest := walk(e.to); rest != nil {
				return append([]string{step}, rest...)
			}
		}
		return nil
	}
	return walk(from)
}

func describeWait(waiter SessionID, key string, holder SessionID) string {
	return string(waiter) + " waits for " + key + " held by " + string(holder)
}

// normalizeClaimKey cleans a path key to a slash-separated canonical form (CL-1). Symbol keys
// ("file|name|line") and hit refs ("file#name@line") are used verbatim, as are other strings,
// which cleaning leaves unchanged unless they contain path separators or dot segments.
func normalizeClaimKey(raw string) (string, error) {
	key := strings.TrimSpace(raw)
	if key == "" {
		return "", errs.NewCode(errs.CodeInvalidInput, "claim key is empty")
	}
	if len(key) > maxClaimKeyBytes {
		return "", errs.NewCode(errs.CodeInvalidInput, "claim key is too long", "bytes", len(key), "max_bytes", maxClaimKeyBytes)
	}
	if strings.ContainsAny(key, "|#") {
		return key, nil
	}
	return filepath.ToSlash(filepath.Clean(key)), nil
}

func observeClaimWait(now time.Time, enqueuedAt string) {
	fn := onClaimWait
	if fn == nil {
		return
	}
	at, err := time.ParseInLocation(time.DateTime, enqueuedAt, time.UTC)
	if err != nil {
		return
	}
	fn(max(0, now.Sub(at)))
}

// nullable stores an empty string as NULL.
func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}
