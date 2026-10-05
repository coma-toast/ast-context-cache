package handoff

import (
	"context"
	"database/sql"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

const (
	// maxTrailEntryTokens caps a live trail entry, text and top hits together (SP-5).
	maxTrailEntryTokens = 60
	// maxTrailTextTokens leaves the rest of the entry's budget for top hits.
	maxTrailTextTokens = 40
	truncationMark     = "…"

	selectOldestTrailQuery = `SELECT id, token_est FROM scratchpad_entries
		WHERE tree_id = ? AND type = '` + string(EntryTypeTrail) + `' ORDER BY id`
	deleteTrailThroughQuery = `DELETE FROM scratchpad_entries
		WHERE tree_id = ? AND type = '` + string(EntryTypeTrail) + `' AND id <= ?`
)

var (
	liveTrailOnce sync.Once
	// liveTrailTarget is the service trail entries are forwarded to.
	liveTrailTarget atomic.Pointer[realService]
)

// liveTrailRefs is a trail entry's refs_json, which Annotate matches siblings' searches against.
type liveTrailRefs struct {
	MatchKey string   `json:"match_key"`
	HitCount int      `json:"hit_count"`
	ZeroHit  bool     `json:"zero_hit"`
	TopHits  []string `json:"top_hits"`
}

// subscribeTrail routes recorded searches to s until ctx is done. trail has no unsubscribe, so
// one process-wide subscription forwards to the newest service rather than each service adding
// its own and every entry being shared once per service ever built.
func (s *realService) subscribeTrail(ctx context.Context) {
	liveTrailOnce.Do(func() { trail.Subscribe(dispatchTrail) })
	liveTrailTarget.Store(s)
	context.AfterFunc(ctx, func() { liveTrailTarget.CompareAndSwap(s, nil) })
}

// onTrail shares a tree session's search with the rest of its tree as a trail entry (SP-5).
// When the tree is at its cap the oldest trail entries make room (RQ-5); if even that can't,
// the entry is dropped, since a search must never fail for want of scratchpad room.
func (s *realService) onTrail(e trail.Entry) {
	if !flags.Enabled(flags.KeyHandoffLiveTrail) {
		return
	}
	sid := SessionID(e.SessionID)
	te, ok := s.trees.lookup(sid)
	if !ok {
		return
	}
	text, refsJSON, tokens := liveTrailEntry(e)
	err := db.HandoffTx(func(tx *sql.Tx) error {
		_, err := s.insertAutoEntryTx(tx, te.tree, sid, EntryTypeTrail, text, refsJSON, tokens)
		return err
	})
	if err != nil {
		s.logger.Warn("Failed to share search with handoff tree", sid.Attr(), te.tree.Attr(), "error", err)
	}
}

// insertAutoEntryTx inserts an entry the server writes on an agent's behalf (trail, claim),
// evicting the oldest trail entries when the tree is at its cap (RQ-5). It returns 0 without an
// error when there is still no room.
func (s *realService) insertAutoEntryTx(tx *sql.Tx, tree TreeID, author SessionID, typ EntryType, text, refsJSON string, tokens int) (int64, error) {
	if err := s.evictTrailTx(tx, tree, tokens, 1); err != nil {
		return 0, err
	}
	err := s.chargeTreeTx(tx, tree, tokens, 1)
	if errs.HasCode(err, CodeHandoffTreeLimitExceeded) {
		s.logger.Debug("Dropped automatic scratchpad entry; tree at its cap", tree.Attr(), author.Attr(), "type", string(typ))
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	return insertEntryTx(tx, tree, author, typ, text, refsJSON, tokens)
}

// evictTrailTx deletes tree's oldest trail entries until tokens and entries more fit under its
// caps, crediting the tree for them. It deletes nothing when evicting every trail entry would
// still not make room.
func (s *realService) evictTrailTx(tx *sql.Tx, tree TreeID, tokens, entries int) error {
	u, err := treeUsageTx(tx, tree)
	if err != nil {
		return err
	}
	lim := LoadLimits()
	needTokens := u.tokens + tokens - lim.TreeMaxTokens
	needEntries := u.entries + entries - lim.TreeMaxEntries
	if needTokens <= 0 && needEntries <= 0 {
		return nil
	}
	rows, err := tx.Query(selectOldestTrailQuery, string(tree))
	if err != nil {
		return errs.WrapMessage("failed to list trail entries to evict", err, "tree", string(tree))
	}
	var lastID int64
	freedTokens, freedEntries := 0, 0
	for rows.Next() {
		if freedTokens >= needTokens && freedEntries >= needEntries {
			break
		}
		var tok int
		if err := rows.Scan(&lastID, &tok); err != nil {
			rows.Close()
			return errs.WrapMessage("failed to read trail entry to evict", err, "tree", string(tree))
		}
		freedTokens += tok
		freedEntries++
	}
	rows.Close()
	if freedTokens < needTokens || freedEntries < needEntries {
		return nil
	}
	if _, err := tx.Exec(deleteTrailThroughQuery, string(tree), lastID); err != nil {
		return errs.WrapMessage("failed to evict trail entries", err, "tree", string(tree))
	}
	s.logger.Debug("Evicted oldest trail entries", tree.Attr(), "entries", freedEntries, "tokens", freedTokens)
	return s.chargeTreeTx(tx, tree, -freedTokens, -freedEntries)
}

// liveTrailEntry formats e as a trail entry of at most maxTrailEntryTokens: the text names the
// tool, normalized query, and hit count, and refs_json carries the match key and top hits.
func liveTrailEntry(e trail.Entry) (text, refsJSON string, tokens int) {
	query := e.QueryNorm
	if query == "" {
		query = trail.NormalizeQuery(e.Query)
	}
	text = trailEntryText(e.Tool, query, e.HitCount)
	if over := len(text) - maxTrailTextTokens*4; over > 0 {
		query = truncateBytes(query, len(query)-over-len(truncationMark)) + truncationMark
		text = trailEntryText(e.Tool, query, e.HitCount)
	}
	tokens = db.EstimateTokens(text)
	hits, hitText := []string{}, ""
	for _, h := range e.TopHits {
		next := strings.TrimSpace(hitText + " " + h)
		if len(hits) == trail.MaxTopHits || tokens+db.EstimateTokens(next) > maxTrailEntryTokens {
			break
		}
		hits, hitText = append(hits, h), next
	}
	tokens += db.EstimateTokens(hitText)
	data, _ := json.Marshal(liveTrailRefs{MatchKey: e.MatchKey(), HitCount: e.HitCount, ZeroHit: e.ZeroHit, TopHits: hits})
	return text, string(data), tokens
}

func dispatchTrail(e trail.Entry) {
	if s := liveTrailTarget.Load(); s != nil {
		s.onTrail(e)
	}
}

// truncateBytes cuts s to at most n bytes without splitting a rune.
func truncateBytes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}
