package handoff

import (
	"context"
	"database/sql"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// trailEntry builds an entry as trail.Record would hand it to subscribers.
func trailEntry(sid SessionID, tool, query string, hits int, topHits ...string) trail.Entry {
	return trail.Entry{
		SessionID: string(sid), Tool: tool, Query: query, QueryNorm: trail.NormalizeQuery(query),
		HitCount: hits, ZeroHit: hits == 0, TopHits: topHits,
	}
}

type trailRow struct {
	id     int64
	author SessionID
	text   string
	refs   liveTrailRefs
	tokens int
}

func trailRows(t *testing.T, tree TreeID) []trailRow {
	t.Helper()
	rows, err := db.ContextDB.Query(`SELECT id, author_session_id, text, refs_json, token_est FROM scratchpad_entries
		WHERE tree_id = ? AND type = 'trail' ORDER BY id`, tree)
	require.NoError(t, err)
	defer rows.Close()
	var out []trailRow
	for rows.Next() {
		var r trailRow
		var refs string
		require.NoError(t, rows.Scan(&r.id, &r.author, &r.text, &refs, &r.tokens))
		require.NoError(t, json.Unmarshal([]byte(refs), &r.refs))
		out = append(out, r)
	}
	require.NoError(t, rows.Err())
	return out
}

func setFlag(t *testing.T, key string, on bool) {
	t.Helper()
	require.NoError(t, flags.Set(key, on))
	// The flag snapshot outlives this test's database, so put the default back before it closes.
	t.Cleanup(func() { require.NoError(t, flags.Set(key, true)) })
}

func TestLiveTrailFromRecord(t *testing.T) {
	initHandoffDB(t)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	s := New(ctx, nil).(*realService)
	tt := seedBareTree(t, "root-live", "alpha", "beta")
	a, b := tt.children[0], tt.children[1]
	recorded := trail.Entry{
		SessionID: string(a), Tool: "search_semantic", Query: "Retry   Backoff", FiltersKey: "lang=go",
		HitCount: 3, TopHits: []string{"retry.go#Retry@10", "backoff.go#Backoff@4"},
	}
	trail.Record(recorded)
	trail.Record(trail.Entry{SessionID: "loner", Tool: "search_semantic", Query: "retry backoff", HitCount: 1})

	// AC17: A's search shows up in the tree's scratchpad in the shared row format.
	rows := trailRows(t, tt.tree)
	require.Len(t, rows, 1, "only tree sessions share their trail")
	r := rows[0]
	assert.Equal(t, a, r.author)
	assert.Equal(t, "search_semantic: retry backoff (3 hits)", r.text)
	recorded.QueryNorm = trail.NormalizeQuery(recorded.Query)
	assert.Equal(t, liveTrailRefs{MatchKey: recorded.MatchKey(), HitCount: 3, TopHits: recorded.TopHits}, r.refs)
	assert.LessOrEqual(t, r.tokens, maxTrailEntryTokens)
	tokens, entries := usage(t, tt.tree)
	assert.Equal(t, 1, entries)
	assert.Equal(t, r.tokens, tokens)
	got, err := s.Read(ctx, ReadRequest{SessionID: b})
	require.NoError(t, err)
	require.Len(t, got.Entries, 1)
	assert.Equal(t, EntryTypeTrail, got.Entries[0].Type)
	assert.Equal(t, recorded.TopHits, got.Entries[0].Refs, "a trail entry's top hits serve as its refs")

	setFlag(t, flags.KeyHandoffLiveTrail, false)
	trail.Record(trail.Entry{SessionID: string(b), Tool: "search_semantic", Query: "off", HitCount: 0})
	assert.Len(t, trailRows(t, tt.tree), 1, "feature_handoff_live_trail off shares nothing")
}

func TestLiveTrailEntryFormat(t *testing.T) {
	t.Parallel()
	hits := []string{
		"a/very/long/path/one.go#First@1", "a/very/long/path/two.go#Second@2", "three.go#Third@3",
		"four.go#Four@4", "five.go#Five@5", "six.go#Six@6",
	}
	text, refs, tokens := liveTrailEntry(trailEntry("s", "get_context_capsule", strings.Repeat("Löng query ", 40), 7, hits...))
	assert.LessOrEqual(t, tokens, maxTrailEntryTokens)
	assert.LessOrEqual(t, db.EstimateTokens(text), maxTrailTextTokens)
	assert.True(t, strings.HasPrefix(text, "get_context_capsule: löng query"))
	assert.True(t, strings.HasSuffix(text, "… (7 hits)"), text)
	var r liveTrailRefs
	require.NoError(t, json.Unmarshal([]byte(refs), &r))
	assert.NotEmpty(t, r.TopHits)
	assert.Less(t, len(r.TopHits), len(hits), "top hits are trimmed to fit the entry")
	_, refs, _ = liveTrailEntry(trailEntry("s", "search_semantic", "nothing", 0))
	assert.JSONEq(t, `{"match_key":"search_semantic|nothing||","hit_count":0,"zero_hit":true,"top_hits":[]}`, refs)
}

// TestLiveTrailEvictsAtCap covers AC22 and RQ-5: at the cap, automatic trail entries evict the
// oldest trail entries, while an explicit post fails with the tree's usage.
func TestLiveTrailEvictsAtCap(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	require.NoError(t, db.SetSetting(SettingTreeMaxEntries, "5"))
	tt := seedBareTree(t, "root-cap", "alpha", "beta")
	a, b := tt.children[0], tt.children[1]
	finding := post(t, s, a, EntryTypeFinding, "keep me")
	dead := post(t, s, b, EntryTypeDeadEnd, "keep me too")
	for _, q := range []string{"one", "two", "three"} {
		s.onTrail(trailEntry(a, "search_semantic", q, 1))
	}
	before := trailRows(t, tt.tree)
	require.Len(t, before, 3)
	_, entries := usage(t, tt.tree)
	require.Equal(t, 5, entries, "the tree is at its entry cap")

	s.onTrail(trailEntry(b, "search_semantic", "four", 1))
	after := trailRows(t, tt.tree)
	require.Len(t, after, 3, "the oldest trail entry made room")
	assert.Equal(t, before[1].id, after[0].id)
	assert.Equal(t, "search_semantic: four (1 hits)", after[2].text)
	tokens, entries := usage(t, tt.tree)
	assert.Equal(t, 5, entries)
	sum := finding.TokenEst + dead.TokenEst
	for _, r := range after {
		sum += r.tokens
	}
	assert.Equal(t, sum, tokens, "evicted entries are credited back")

	_, err := s.Post(ctx, PostRequest{SessionID: a, Type: EntryTypeFinding, Text: "one too many"})
	require.True(t, errs.HasCode(err, CodeHandoffTreeLimitExceeded), "%v", err)
	details := ErrorMap(err)["details"].(map[string]any)
	assert.Equal(t, 5, details["entries_used"])
	assert.Equal(t, 5, details["entries_max"])
	assert.Equal(t, tokens, details["tokens_used"])
	assert.Equal(t, defaultTreeMaxTokens, details["tokens_max"])
	assert.Equal(t, 2, count(t, `SELECT COUNT(*) FROM scratchpad_entries WHERE tree_id = ? AND type <> 'trail'`, tt.tree),
		"findings and dead ends are never evicted")

	// With no trail entries left to evict, an automatic entry is dropped, not an error.
	exec(t, `DELETE FROM scratchpad_entries WHERE tree_id = ? AND type = 'trail'`, tt.tree)
	exec(t, `UPDATE handoff_trees SET entries_used = 5 WHERE tree_id = ?`, tt.tree)
	s.onTrail(trailEntry(a, "search_semantic", "five", 1))
	assert.Empty(t, trailRows(t, tt.tree))
}

func TestLiveTrailEvictsForTokens(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	tt := seedBareTree(t, "root-tok", "alpha")
	a := tt.children[0]
	for _, q := range []string{"first query", "second query"} {
		s.onTrail(trailEntry(a, "search_semantic", q, 2))
	}
	rows := trailRows(t, tt.tree)
	tokens, _ := usage(t, tt.tree)
	require.NoError(t, db.SetSetting(SettingTreeMaxTokens, strconv.Itoa(tokens)))
	s.onTrail(trailEntry(a, "search_semantic", "third query", 2))
	after := trailRows(t, tt.tree)
	require.Len(t, after, 2)
	assert.Equal(t, rows[1].id, after[0].id, "the token cap evicts the oldest trail entry too")
	err := db.HandoffTx(func(tx *sql.Tx) error { return s.chargeTreeTx(tx, tt.tree, 1, 0) })
	assert.True(t, errs.HasCode(err, CodeHandoffTreeLimitExceeded))
	err = db.HandoffTx(func(tx *sql.Tx) error { return s.chargeTreeTx(tx, "hft_0000000000000000", 1, 1) })
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound))
}
