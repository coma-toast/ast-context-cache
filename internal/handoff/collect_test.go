package handoff

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

func complete(t *testing.T, s *realService, sid SessionID, status Status, summary string) *CompleteResponse {
	t.Helper()
	res, err := s.Complete(context.Background(), CompleteRequest{SessionID: sid, Content: "result of " + string(sid), Summary: summary, Status: status})
	require.NoError(t, err)
	return res
}

func childSessions(children []ChildResult) []SessionID {
	out := make([]SessionID, len(children))
	for i, c := range children {
		out[i] = c.SessionID
	}
	return out
}

func TestCollectMixedStatuses(t *testing.T) {
	s := newFanInService(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	h := seedHandoff(t, handoffSeed{root: "parent", label: "fan", children: 16, at: now})
	fullSummary := tokensOf(LoadLimits().SummaryMaxTokens)
	results := map[SessionID]*CompleteResponse{}
	for _, c := range h.children[:10] {
		results[c] = complete(t, s, c, StatusDone, fullSummary)
	}
	for _, c := range h.children[10:12] {
		results[c] = complete(t, s, c, StatusFailed, "boom")
	}
	_, err := s.Complete(context.Background(), CompleteRequest{
		SessionID: h.children[0], Content: "again", Summary: fullSummary,
		ChangedFiles: []string{"a.go"}, OpenQuestions: []string{"why?"},
	})
	require.NoError(t, err)
	_, err = contextnotes.Store(string(h.children[12]), "partial work", "wip", "/p", nil, "", nil, nil)
	require.NoError(t, err)
	exec(t, `INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES (?, 'x.go', ?)`, h.tree, h.children[13])
	exec(t, `UPDATE handoff_children SET last_activity_at = ? WHERE child_session_id = ?`, sqlTime(now.Add(-31*time.Minute)), h.children[15])
	_, err = s.markAbandoned()
	require.NoError(t, err)

	res, err := s.Collect(context.Background(), CollectRequest{Handoff: h.ref})
	require.NoError(t, err)
	require.Len(t, res.Children, 16, "AC12: every child in one response")
	assert.False(t, res.Truncated)
	lim := LoadLimits()
	assert.LessOrEqual(t, res.TokensUsed, lim.MaxChildren*(lim.SummaryMaxTokens+collectEntryTokens))
	counts := map[Status]int{}
	for i, c := range res.Children {
		counts[c.Status]++
		assert.Equal(t, h.children[i], c.SessionID, "children in open order")
		assert.Equal(t, h.ref, c.Handoff)
		assert.Equal(t, "fan", c.Label, "a child's label falls back to its handoff's")
		assert.Equal(t, 1, c.Depth)
		assert.NotEmpty(t, c.LastActivityAt)
	}
	assert.Equal(t, map[Status]int{StatusDone: 10, StatusFailed: 2, StatusOpen: 3, StatusAbandoned: 1}, counts)
	first := res.Children[0]
	assert.NotEqual(t, results[h.children[0]].ResultRef, first.ResultRef, "the current result")
	assert.Equal(t, 2, first.NoteCount, "both results are the child's notes")
	assert.Equal(t, []string{"a.go"}, first.ChangedFiles, "RT-8 fields echoed")
	assert.Equal(t, []string{"why?"}, first.OpenQuestions)
	assert.False(t, first.SummaryTruncated, "a summary exactly at the cap is kept whole")
	failed := res.Children[10]
	assert.Equal(t, "boom", failed.Summary)
	assert.Equal(t, results[h.children[10]].ResultRef, failed.ResultRef)
	assert.Equal(t, 1, res.Children[12].NoteCount, "an open child's partial work is visible (FI-4)")
	assert.Empty(t, res.Children[12].ResultRef)
	assert.Equal(t, 1, res.Children[13].ActiveClaims)
	assert.Equal(t, StatusAbandoned, res.Children[15].Status)

	bySession, err := s.Collect(context.Background(), CollectRequest{SessionID: h.root})
	require.NoError(t, err)
	assert.Equal(t, res.Children, bySession.Children, "by parent session: all its handoffs")

	small, err := s.Collect(context.Background(), CollectRequest{Handoff: h.ref, TokenBudget: 500})
	require.NoError(t, err)
	assert.True(t, small.Truncated)
	assert.NotEmpty(t, small.Children)
	assert.Less(t, len(small.Children), 16)
	assert.LessOrEqual(t, small.TokensUsed, 500)
}

func TestCollectRecursive(t *testing.T) {
	s := newFanInService(t)
	top := seedHandoff(t, handoffSeed{root: "parent", children: 2})
	nested := seedHandoff(t, handoffSeed{root: "parent", tree: top.tree, parentChild: top.children[0], depth: 2, label: "deep", children: 2})
	other := seedHandoff(t, handoffSeed{root: "someone-else", children: 1})
	tests := []struct {
		name string
		req  CollectRequest
		want []SessionID
	}{
		{"direct children", CollectRequest{Handoff: top.ref}, top.children},
		{
			"subtree depth first",
			CollectRequest{Handoff: top.ref, Recursive: true},
			[]SessionID{top.children[0], nested.children[0], nested.children[1], top.children[1]},
		},
		{"nested node", CollectRequest{SessionID: top.children[0]}, nested.children},
		{"root session", CollectRequest{SessionID: "parent"}, top.children},
		{"other parent", CollectRequest{SessionID: "someone-else"}, other.children},
		{"no handoffs", CollectRequest{SessionID: "nobody"}, []SessionID{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := s.Collect(context.Background(), tt.req)
			require.NoError(t, err)
			assert.Equal(t, tt.want, childSessions(res.Children))
		})
	}
	res, err := s.Collect(context.Background(), CollectRequest{Handoff: top.ref, Recursive: true})
	require.NoError(t, err)
	assert.Equal(t, 2, res.Children[1].Depth)
	assert.Equal(t, "deep", res.Children[1].Label)

	_, err = s.Collect(context.Background(), CollectRequest{})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
	_, err = s.Collect(context.Background(), CollectRequest{Handoff: "hof_0000000000000000"})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "%v", err)
	_, err = s.Collect(context.Background(), CollectRequest{Handoff: "nope"})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
}

func TestCollectWait(t *testing.T) {
	s := newFanInService(t)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 2})
	ctx := context.Background()

	t.Run("returns on status change", func(t *testing.T) {
		go func() {
			time.Sleep(100 * time.Millisecond)
			_, err := s.Complete(ctx, CompleteRequest{SessionID: h.children[0], Content: "done", Summary: "finished"})
			assert.NoError(t, err)
		}()
		start := time.Now()
		res, err := s.Collect(ctx, CollectRequest{Handoff: h.ref, WaitSeconds: 30})
		require.NoError(t, err)
		assert.Less(t, time.Since(start), 10*time.Second)
		assert.True(t, res.Waited)
		assert.Equal(t, StatusDone, res.Children[0].Status, "the response reflects the change")
	})
	t.Run("times out", func(t *testing.T) {
		start := time.Now()
		res, err := s.Collect(ctx, CollectRequest{Handoff: h.ref, WaitSeconds: 1})
		require.NoError(t, err)
		assert.GreaterOrEqual(t, time.Since(start), time.Second)
		assert.True(t, res.Waited)
	})
	t.Run("context canceled", func(t *testing.T) {
		cctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
		defer cancel()
		_, err := s.Collect(cctx, CollectRequest{Handoff: h.ref, WaitSeconds: 60})
		assert.ErrorIs(t, err, context.DeadlineExceeded)
	})
	t.Run("wakes across trees", func(t *testing.T) {
		second := seedHandoff(t, handoffSeed{root: "parent", children: 1})
		go func() {
			time.Sleep(100 * time.Millisecond)
			_, err := s.Complete(ctx, CompleteRequest{SessionID: second.children[0], Content: "done", Summary: "other tree"})
			assert.NoError(t, err)
		}()
		start := time.Now()
		res, err := s.Collect(ctx, CollectRequest{SessionID: "parent", WaitSeconds: 30})
		require.NoError(t, err)
		assert.Less(t, time.Since(start), 10*time.Second)
		assert.True(t, res.Waited)
	})
	t.Run("nothing open returns at once", func(t *testing.T) {
		complete(t, s, h.children[1], StatusFailed, "gave up")
		start := time.Now()
		res, err := s.Collect(ctx, CollectRequest{Handoff: h.ref, WaitSeconds: 60})
		require.NoError(t, err)
		assert.Less(t, time.Since(start), time.Second)
		assert.False(t, res.Waited)
	})
}

func TestList(t *testing.T) {
	s := newFanInService(t)
	older := seedHandoff(t, handoffSeed{root: "parent", label: "first", children: 2, at: time.Now().Add(-time.Hour)})
	newer := seedHandoff(t, handoffSeed{root: "parent", tree: older.tree, label: "second", children: 1})
	seedHandoff(t, handoffSeed{root: "someone-else", children: 1})
	seedHandoff(t, handoffSeed{root: "parent", tree: older.tree, parentChild: older.children[0], depth: 2, children: 1})
	complete(t, s, older.children[0], StatusDone, "ok")

	res, err := s.List(context.Background(), ListRequest{SessionID: "parent"})
	require.NoError(t, err)
	require.Len(t, res.Handoffs, 2, "AC13: the parent's own handoffs, not its children's")
	assert.Equal(t, newer.ref, res.Handoffs[0].Ref, "newest first")
	assert.Equal(t, "second", res.Handoffs[0].Label)
	assert.Equal(t, map[Status]int{StatusOpen: 1}, res.Handoffs[0].StatusCounts)
	got := res.Handoffs[1]
	assert.Equal(t, older.ref, got.Ref)
	assert.Equal(t, older.tree, got.TreeID)
	assert.Equal(t, ModeFresh, got.Mode)
	assert.Equal(t, 1, got.Depth)
	assert.Equal(t, 2, got.Children)
	assert.Equal(t, map[Status]int{StatusOpen: 1, StatusDone: 1}, got.StatusCounts)

	res, err = s.List(context.Background(), ListRequest{SessionID: "nobody"})
	require.NoError(t, err)
	assert.Empty(t, res.Handoffs)
	_, err = s.List(context.Background(), ListRequest{})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
}

func TestStatus(t *testing.T) {
	s := newFanInService(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 2, at: now})
	complete(t, s, h.children[0], StatusDone, "ok")
	exec(t, `INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES (?, 'x.go', ?)`, h.tree, h.children[1])
	exec(t, `INSERT INTO handoff_claim_queue (tree_id, key, session_id) VALUES (?, 'x.go', 'w1'), (?, 'x.go', 'w2')`, h.tree, h.tree)
	for _, req := range []StatusRequest{{TreeID: h.tree}, {Handoff: h.ref}, {SessionID: h.children[1]}, {SessionID: h.root}} {
		res, err := s.Status(context.Background(), req)
		require.NoError(t, err, "%+v", req)
		assert.Equal(t, h.tree, res.TreeID)
		assert.Equal(t, h.root, res.RootSessionID)
		assert.Equal(t, "/p", res.ProjectPath)
		assert.Equal(t, sqlTime(now), res.LastAccessAt)
		assert.Equal(t, sqlTime(now.Add(7*24*time.Hour)), res.ExpiresAt)
		assert.Equal(t, db.EstimateTokens("result of "+string(h.children[0])), res.TokensUsed)
		assert.Equal(t, 1, res.EntriesUsed)
		assert.Equal(t, 64000, res.TokensMax)
		assert.Equal(t, 300, res.EntriesMax)
		require.Len(t, res.Handoffs, 1)
		assert.Equal(t, map[Status]int{StatusOpen: 1, StatusDone: 1}, res.Handoffs[0].StatusCounts)
		assert.Equal(t, 1, res.ActiveClaims)
		assert.Equal(t, 2, res.QueuedClaims)
	}
	tests := []struct {
		name string
		req  StatusRequest
		code errs.Code
	}{
		{"nothing named", StatusRequest{}, errs.CodeInvalidInput},
		{"bad tree id", StatusRequest{TreeID: "tree"}, errs.CodeInvalidInput},
		{"unknown tree", StatusRequest{TreeID: "hft_0000000000000000"}, CodeHandoffNotFound},
		{"unknown handoff", StatusRequest{Handoff: "hof_0000000000000000"}, CodeHandoffNotFound},
		{"unknown session", StatusRequest{SessionID: "nobody"}, CodeHandoffNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.Status(context.Background(), tt.req)
			assert.True(t, errs.HasCode(err, tt.code), "%v", err)
		})
	}
}

func TestFlush(t *testing.T) {
	s := newFanInService(t)
	ctx := context.Background()
	byRef := seedHandoff(t, handoffSeed{root: "p1", children: 2})
	complete(t, s, byRef.children[0], StatusDone, "ok")
	res, err := s.Flush(ctx, FlushRequest{Handoff: byRef.ref})
	require.NoError(t, err)
	assert.Equal(t, &FlushResponse{TreeID: byRef.tree, Handoffs: 1, Children: 2, NotesDeleted: 1}, res)
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_trees WHERE tree_id = ?`, byRef.tree))
	_, err = s.Flush(ctx, FlushRequest{Handoff: byRef.ref})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "%v", err)

	a := seedHandoff(t, handoffSeed{root: "p2", children: 1, at: time.Now().Add(-time.Hour)})
	b := seedHandoff(t, handoffSeed{root: "p2", children: 1})
	_, err = s.Flush(ctx, FlushRequest{SessionID: a.children[0]})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "a child flushes by ref, not by session: %v", err)
	res, err = s.Flush(ctx, FlushRequest{SessionID: "p2"})
	require.NoError(t, err)
	assert.Equal(t, &FlushResponse{TreeID: b.tree, Handoffs: 2, Children: 2}, res, "every tree the root started")
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_trees WHERE root_session_id = 'p2'`))

	c := seedHandoff(t, handoffSeed{root: "p3", children: 1})
	res, err = s.Flush(ctx, FlushRequest{TreeID: c.tree})
	require.NoError(t, err)
	assert.Equal(t, c.tree, res.TreeID)
	_, err = s.Flush(ctx, FlushRequest{TreeID: c.tree})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "%v", err)
	_, err = s.Flush(ctx, FlushRequest{})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
}

// TestSweepRemovesExpiredTreeData is AC21: a tree idle past the TTL loses every row, its
// children's notes, dedup rows, and search trail, while the parent keeps its own.
func TestSweepRemovesExpiredTreeData(t *testing.T) {
	s := newFanInService(t)
	start := time.Now()
	h := seedHandoff(t, handoffSeed{root: "parent", children: 2, at: start})
	child := h.children[0]
	complete(t, s, child, StatusDone, "ok")
	_, err := contextnotes.Store(string(child), "working notes", "", "/p", nil, "", nil, nil)
	require.NoError(t, err)
	astcontext.MarkReturned(string(child), astcontext.ReturnedSymbol{File: "/p/a.go", Name: "A", StartLine: 1})
	for _, sid := range []SessionID{child, h.children[1], h.root} {
		trail.Record(trail.Entry{SessionID: string(sid), Tool: "search_semantic", Query: "retry " + string(sid), HitCount: 2})
	}
	db.FlushWriteBuffers()
	trail.Record(trail.Entry{SessionID: string(child), Tool: "search_semantic", Query: "still buffered", HitCount: 1})

	setClock(t, start.Add(7*24*time.Hour+time.Hour))
	n, err := s.sweep()
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	assert.Zero(t, count(t, `SELECT
		(SELECT COUNT(*) FROM handoff_trees WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoffs WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_children WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_results WHERE child_session_id LIKE ?2 || '.c%')`, h.tree, h.ref))
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM context_notes WHERE session_id LIKE ? || '.c%'`, h.ref), "child notes and results")
	assert.Empty(t, astcontext.ReturnedKeys(string(child)), "child dedup rows")
	db.FlushWriteBuffers()
	var childTrail, parentTrail int
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM search_trail WHERE session_id LIKE ? || '.c%'`, string(h.ref)).Scan(&childTrail))
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM search_trail WHERE session_id = ?`, string(h.root)).Scan(&parentTrail))
	assert.Zero(t, childTrail, "child search trail, including rows still buffered at the sweep")
	assert.Empty(t, trail.ForSession(string(child), 10))
	assert.Equal(t, 1, parentTrail, "the parent's own trail stays until it ages out")
	_, err = s.Collect(context.Background(), CollectRequest{Handoff: h.ref})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "%v", err)
}
