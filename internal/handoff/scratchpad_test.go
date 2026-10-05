package handoff

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// seedBareTree inserts a tree with one handoff and a child per label, with no scratchpad
// entries, claims, or usage, standing in for Create and Open.
func seedBareTree(t testing.TB, root SessionID, labels ...string) testTree {
	t.Helper()
	tree, err := NewTreeID()
	require.NoError(t, err)
	ref, err := NewHandoffRef()
	require.NoError(t, err)
	ts := sqlTime(nowFunc())
	tt := testTree{tree: tree, root: root, ref: ref}
	exec(t, `INSERT INTO handoff_trees (tree_id, root_session_id, project_path, created_at, last_access_at) VALUES (?, ?, '/p', ?, ?)`, tree, root, ts, ts)
	exec(t, `INSERT INTO handoffs (ref, tree_id, parent_session_id, depth, mode, brief, project_path, child_count, created_at, last_access_at)
		VALUES (?, ?, ?, 1, 'fresh', 'do it', '/p', ?, ?, ?)`, ref, tree, root, len(labels), ts, ts)
	for i, label := range labels {
		child := ChildSessionID(ref, i+1)
		tt.children = append(tt.children, child)
		exec(t, `INSERT INTO handoff_children (child_session_id, handoff_ref, tree_id, label, status, project_path, opened_at, last_activity_at)
			VALUES (?, ?, ?, ?, 'open', '/p', ?, ?)`, child, ref, tree, label, ts, ts)
	}
	return tt
}

func initHandoffDB(t testing.TB) {
	t.Helper()
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)
}

func usage(t *testing.T, tree TreeID) (tokens, entries int) {
	t.Helper()
	require.NoError(t, db.ContextDB.QueryRow(`SELECT tokens_used, entries_used FROM handoff_trees WHERE tree_id = ?`, tree).Scan(&tokens, &entries))
	return tokens, entries
}

func entryIDs(entries []ScratchpadEntry) []int64 {
	out := make([]int64, len(entries))
	for i, e := range entries {
		out[i] = e.ID
	}
	return out
}

func post(t *testing.T, s *realService, sid SessionID, typ EntryType, text string, refs ...string) *PostResponse {
	t.Helper()
	res, err := s.Post(context.Background(), PostRequest{SessionID: sid, Type: typ, Text: text, Refs: refs})
	require.NoError(t, err)
	return res
}

func TestPostAndReadSinceCursor(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	tt := seedBareTree(t, "root-sp", "alpha", "beta")
	a, b := tt.children[0], tt.children[1]
	own := post(t, s, b, EntryTypeFinding, "b's own note")
	first, err := s.Read(ctx, ReadRequest{SessionID: b})
	require.NoError(t, err)
	assert.Empty(t, first.Entries, "own entries are excluded by default")
	assert.Equal(t, int64(0), first.NextCursor)

	found := post(t, s, a, EntryTypeFinding, "  retry lives in backoff.go  ", "internal/backoff.go", "ctx_abc", "ctx_abc", " ")
	assert.Equal(t, tt.tree, found.TreeID)
	assert.Greater(t, found.ID, own.ID)
	tokens, entries := usage(t, tt.tree)
	assert.Equal(t, 2, entries, "every post is charged to the tree (RQ-4)")
	assert.Equal(t, own.TokenEst+found.TokenEst, tokens)
	assert.Equal(t, tokens, found.TokensUsed)
	assert.Equal(t, entries, found.EntriesUsed)

	// AC16: B, reading from its last cursor, gets exactly A's finding.
	got, err := s.Read(ctx, ReadRequest{SessionID: b, Since: first.NextCursor})
	require.NoError(t, err)
	require.Len(t, got.Entries, 1)
	e := got.Entries[0]
	assert.Equal(t, found.ID, e.ID)
	assert.Equal(t, a, e.Author)
	assert.Equal(t, EntryTypeFinding, e.Type)
	assert.Equal(t, "retry lives in backoff.go", e.Text)
	assert.Equal(t, []string{"internal/backoff.go", "ctx_abc"}, e.Refs)
	assert.NotEmpty(t, e.CreatedAt)
	assert.Equal(t, found.ID, got.NextCursor)
	assert.False(t, got.Truncated)
	again, err := s.Read(ctx, ReadRequest{SessionID: b, Since: got.NextCursor})
	require.NoError(t, err)
	assert.Empty(t, again.Entries, "nothing new after the cursor")

	all, err := s.Read(ctx, ReadRequest{SessionID: b, IncludeOwn: true})
	require.NoError(t, err)
	assert.Equal(t, []int64{own.ID, found.ID}, entryIDs(all.Entries))
	mine, err := s.Read(ctx, ReadRequest{SessionID: b, Author: b})
	require.NoError(t, err)
	assert.Equal(t, []int64{own.ID}, entryIDs(mine.Entries), "naming yourself as author includes your entries")
	dead := post(t, s, a, EntryTypeDeadEnd, "grepping for Retry finds nothing")
	byType, err := s.Read(ctx, ReadRequest{SessionID: tt.root, Types: []EntryType{EntryTypeFinding}})
	require.NoError(t, err)
	assert.Equal(t, []int64{own.ID, found.ID}, entryIDs(byType.Entries), "the root reads its tree too")
	byAuthor, err := s.Read(ctx, ReadRequest{SessionID: tt.root, Author: a})
	require.NoError(t, err)
	assert.Equal(t, []int64{found.ID, dead.ID}, entryIDs(byAuthor.Entries))
	rootPost := post(t, s, tt.root, EntryTypeFinding, "root can post")
	assert.NotZero(t, rootPost.ID)
}

func TestPostRejects(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	tt := seedBareTree(t, "root-rej", "alpha")
	child := tt.children[0]
	manyRefs := make([]string, maxEntryRefs+1)
	for i := range manyRefs {
		manyRefs[i] = "ref" + strconv.Itoa(i)
	}
	tests := []struct {
		name string
		req  PostRequest
		code errs.Code
	}{
		{"outside a tree", PostRequest{SessionID: "loner", Type: EntryTypeFinding, Text: "x"}, CodeHandoffNotFound},
		{"claim type", PostRequest{SessionID: child, Type: EntryTypeClaim, Text: "x"}, errs.CodeInvalidInput},
		{"trail type", PostRequest{SessionID: child, Type: EntryTypeTrail, Text: "x"}, errs.CodeInvalidInput},
		{"empty text", PostRequest{SessionID: child, Type: EntryTypeFinding, Text: "  "}, errs.CodeInvalidInput},
		{"over 500 tokens", PostRequest{SessionID: child, Type: EntryTypeFinding, Text: strings.Repeat("x", 2004)}, errs.CodeInvalidInput},
		{"too many refs", PostRequest{SessionID: child, Type: EntryTypeFinding, Text: "x", Refs: manyRefs}, errs.CodeInvalidInput},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := s.Post(context.Background(), tc.req)
			assert.True(t, errs.HasCode(err, tc.code), "%v", err)
		})
	}
	post(t, s, child, EntryTypeFinding, strings.Repeat("x", 2000))
	_, err := s.Read(context.Background(), ReadRequest{SessionID: child, Types: []EntryType{"bogus"}})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	_, err = s.Read(context.Background(), ReadRequest{SessionID: "loner"})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound))
}

func TestReadPagesWithinBudget(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	tt := seedBareTree(t, "root-page", "alpha", "beta")
	var ids []int64
	for range 5 {
		ids = append(ids, post(t, s, tt.children[0], EntryTypeFinding, strings.Repeat("y", 160)).ID) // 40 tokens + 10 overhead
	}
	page, err := s.Read(ctx, ReadRequest{SessionID: tt.children[1], TokenBudget: 120})
	require.NoError(t, err)
	assert.Equal(t, ids[:2], entryIDs(page.Entries))
	assert.True(t, page.Truncated)
	assert.Equal(t, ids[1], page.NextCursor)
	assert.Equal(t, 100, page.TokensUsed)
	page, err = s.Read(ctx, ReadRequest{SessionID: tt.children[1], TokenBudget: 1, Since: page.NextCursor})
	require.NoError(t, err)
	assert.Equal(t, ids[2:3], entryIDs(page.Entries), "one entry always fits so reads make progress")
	assert.True(t, page.Truncated)
	page, err = s.Read(ctx, ReadRequest{SessionID: tt.children[1], Since: page.NextCursor})
	require.NoError(t, err)
	assert.Equal(t, ids[3:], entryIDs(page.Entries))
	assert.False(t, page.Truncated)
}

func TestRetract(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	tt := seedBareTree(t, "root-retract", "alpha", "beta")
	a, b := tt.children[0], tt.children[1]
	keep := post(t, s, a, EntryTypeFinding, "still true")
	wrong := post(t, s, a, EntryTypeFinding, "turned out wrong")
	_, err := s.Retract(ctx, RetractRequest{SessionID: b, Entry: wrong.ID})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "only the author retracts: %v", err)
	_, err = s.Retract(ctx, RetractRequest{SessionID: a, Entry: 9999})
	assert.True(t, errs.HasCode(err, errs.CodeNotFound))
	_, err = s.Retract(ctx, RetractRequest{SessionID: "loner", Entry: wrong.ID})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound))
	res, err := s.Retract(ctx, RetractRequest{SessionID: a, Entry: wrong.ID})
	require.NoError(t, err)
	assert.Equal(t, &RetractResponse{Entry: wrong.ID, Retracted: true}, res)
	_, err = s.Retract(ctx, RetractRequest{SessionID: a, Entry: wrong.ID})
	require.NoError(t, err, "retracting twice is a no-op")

	got, err := s.Read(ctx, ReadRequest{SessionID: b})
	require.NoError(t, err)
	assert.Equal(t, []int64{keep.ID}, entryIDs(got.Entries), "retracted entries are hidden")
	got, err = s.Read(ctx, ReadRequest{SessionID: b, IncludeRetracted: true})
	require.NoError(t, err)
	require.Len(t, got.Entries, 2)
	assert.True(t, got.Entries[1].Retracted)
	assert.Equal(t, 2, count(t, `SELECT COUNT(*) FROM scratchpad_entries WHERE tree_id = ?`, tt.tree), "retraction does not delete")
}

func TestReadDeadEndsView(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	tt := seedBareTree(t, "root-dead", "alpha", "beta")
	a, b := tt.children[0], tt.children[1]
	item := func(ord int, content snapshotTrailJSON) {
		data, err := json.Marshal(content)
		require.NoError(t, err)
		exec(t, `INSERT INTO handoff_snapshot_items (handoff_ref, section, ord, item_key, label, content, token_est) VALUES (?, 'trail', ?, ?, ?, ?, 5)`,
			tt.ref, ord, content.Tool+"|"+content.Query+"||", content.Query, string(data))
	}
	item(0, snapshotTrailJSON{Tool: "search_semantic", Query: "legacy retry", ZeroHit: true})
	item(1, snapshotTrailJSON{Tool: "search_semantic", Query: "backoff", HitCount: 4, TopHits: []string{"b.go#B@1"}})
	finding := post(t, s, a, EntryTypeFinding, "a finding")
	deadA := post(t, s, a, EntryTypeDeadEnd, "no retry in vendor/")
	post(t, s, b, EntryTypeDeadEnd, "b's own dead end")
	s.onTrail(trailEntry(a, "get_context_capsule", "RetryPolicy", 0))
	s.onTrail(trailEntry(a, "get_context_capsule", "Backoff", 2))
	zeroID := int64(count(t, `SELECT id FROM scratchpad_entries WHERE tree_id = ? AND text LIKE '%retrypolicy%'`, tt.tree))

	got, err := s.Read(ctx, ReadRequest{SessionID: b, Types: []EntryType{EntryTypeDeadEnd}})
	require.NoError(t, err)
	assert.Equal(t, []int64{deadA.ID}, entryIDs(got.Entries), "the type filter pages dead_end posts only")
	require.Len(t, got.DeadEnds, 2, "zero-hit trail entries and snapshot zero-hit searches, minus entries already in the page and b's own")
	assert.Equal(t, zeroID, got.DeadEnds[0].ID)
	assert.Equal(t, "get_context_capsule: retrypolicy (0 hits)", got.DeadEnds[0].Text)
	assert.Equal(t, int64(0), got.DeadEnds[1].ID, "snapshot items have no scratchpad id")
	assert.Equal(t, tt.root, got.DeadEnds[1].Author)
	assert.Equal(t, "search_semantic: legacy retry (0 hits)", got.DeadEnds[1].Text)

	got, err = s.Read(ctx, ReadRequest{SessionID: b, Types: []EntryType{EntryTypeDeadEnd}, Since: deadA.ID})
	require.NoError(t, err)
	assert.Empty(t, got.Entries)
	assert.Len(t, got.DeadEnds, 3, "the view does not depend on the cursor")
	plain, err := s.Read(ctx, ReadRequest{SessionID: b, Since: finding.ID - 1})
	require.NoError(t, err)
	assert.Nil(t, plain.DeadEnds, "only reads asking for dead ends get the view")
	root, err := s.Read(ctx, ReadRequest{SessionID: tt.root, Types: []EntryType{EntryTypeDeadEnd}})
	require.NoError(t, err)
	assert.Len(t, root.DeadEnds, 1, "a root opened no snapshot; its page already holds both dead_end posts")
}
