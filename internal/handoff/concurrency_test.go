package handoff

import (
	"context"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// NFR-4 under make race. Sixteen concurrent claims on one key (AC20: one holder, fifteen
// waiters in arrival order, FIFO grants) are TestClaimConcurrentFIFO in claims_test.go; the
// tests here cover concurrent opens, scratchpad posts against cursor reads, complete against
// collect, and a whole tree opening, posting, claiming, and searching at once.

// concurrently runs fn(i) for i in [0, n) on n goroutines released together, and waits.
func concurrently(n int, fn func(i int)) {
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() {
			<-start
			fn(i)
		})
	}
	close(start)
	wg.Wait()
}

// TestConcurrentOpens covers OP-1 and AC14 under contention: sixteen simultaneous opens mint
// sixteen distinct children, and the cap holds however many race for it.
func TestConcurrentOpens(t *testing.T) {
	s := newTestService(t)
	ctx := context.Background()
	created := mustCreate(t, s, CreateRequest{SessionID: "root-opens", ProjectPath: "/p"})
	opened := make([]*OpenResponse, defaultMaxChildren)
	openErrs := make([]error, defaultMaxChildren)
	concurrently(defaultMaxChildren, func(i int) {
		opened[i], openErrs[i] = s.Open(ctx, OpenRequest{Handoff: created.Ref})
	})
	seen := map[SessionID]bool{}
	for i, res := range opened {
		require.NoError(t, openErrs[i])
		assert.False(t, seen[res.SessionID], "child %s minted twice", res.SessionID)
		seen[res.SessionID] = true
	}
	for n := 1; n <= defaultMaxChildren; n++ {
		assert.True(t, seen[ChildSessionID(created.Ref, n)], "child %d", n)
	}
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT child_count FROM handoffs WHERE ref = ?`, created.Ref))
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT COUNT(*) FROM handoff_children WHERE handoff_ref = ?`, created.Ref))
	_, err := s.Open(ctx, OpenRequest{Handoff: created.Ref})
	assert.True(t, errs.HasCode(err, CodeHandoffChildrenExceeded), "the 17th open: %v", err)

	// More openers than room: exactly the cap succeed, the rest get the cap error.
	over := mustCreate(t, s, CreateRequest{SessionID: "root-opens-over", ProjectPath: "/p"})
	racers := defaultMaxChildren + 8
	overErrs := make([]error, racers)
	concurrently(racers, func(i int) {
		_, overErrs[i] = s.Open(ctx, OpenRequest{Handoff: over.Ref})
	})
	ok := 0
	for _, err := range overErrs {
		if err == nil {
			ok++
			continue
		}
		assert.True(t, errs.HasCode(err, CodeHandoffChildrenExceeded), "%v", err)
	}
	assert.Equal(t, defaultMaxChildren, ok)
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT COUNT(*) FROM handoff_children WHERE handoff_ref = ?`, over.Ref))
}

// TestConcurrentPostsAndCursorReads covers SP-1 and SP-4 under contention: eight sessions post
// while reading from their own cursors, and every reader sees every other session's entries
// exactly once and never its own.
func TestConcurrentPostsAndCursorReads(t *testing.T) {
	s := newTestService(t)
	ctx := context.Background()
	const sessions, posts = 8, 20
	created := mustCreate(t, s, CreateRequest{SessionID: "root-pad", ProjectPath: "/p"})
	children := make([]SessionID, sessions)
	for i := range children {
		children[i] = mustOpen(t, s, created.Ref, "/p").SessionID
	}
	seen := make([]map[int64]int, sessions)
	postedIDs := make([][]int64, sessions)
	cursors := make([]int64, sessions)
	failures := make([]error, sessions)
	// drain reads from the session's cursor until a page comes back empty.
	drain := func(i int) error {
		for {
			res, err := s.Read(ctx, ReadRequest{SessionID: children[i], Since: cursors[i]})
			if err != nil {
				return err
			}
			for _, e := range res.Entries {
				seen[i][e.ID]++
			}
			cursors[i] = res.NextCursor
			if len(res.Entries) == 0 {
				return nil
			}
		}
	}
	concurrently(sessions, func(i int) {
		seen[i] = map[int64]int{}
		for n := range posts {
			res, err := s.Post(ctx, PostRequest{SessionID: children[i], Type: EntryTypeFinding, Text: "finding " + strconv.Itoa(n) + " from " + strconv.Itoa(i)})
			if err != nil {
				failures[i] = err
				return
			}
			postedIDs[i] = append(postedIDs[i], res.ID)
			if err := drain(i); err != nil {
				failures[i] = err
				return
			}
		}
	})
	for i := range children {
		require.NoError(t, failures[i], "session %d", i)
		require.NoError(t, drain(i), "final drain for session %d", i)
	}
	all := map[int64]int{}
	for i, ids := range postedIDs {
		require.Len(t, ids, posts)
		for _, id := range ids {
			all[id] = i
		}
	}
	require.Len(t, all, sessions*posts, "every post got its own id")
	for i := range children {
		for id, n := range seen[i] {
			assert.Equal(t, 1, n, "session %d read entry %d %d times", i, id, n)
			assert.NotEqual(t, i, all[id], "session %d read its own entry %d", i, id)
		}
		assert.Len(t, seen[i], (sessions-1)*posts, "session %d saw every other session's entry", i)
	}
	tokens, entries := usage(t, created.TreeID)
	assert.Equal(t, sessions*posts, entries, "no lost cap charges")
	assert.Positive(t, tokens)
}

// TestConcurrentCompleteAndCollect covers RT-1 and FI-1 under contention: collects racing
// sixteen completions only ever see open or done children, a done child always with its
// result, never a done child going back to open, and finally all sixteen done.
func TestConcurrentCompleteAndCollect(t *testing.T) {
	s := newTestService(t)
	ctx := context.Background()
	created := mustCreate(t, s, CreateRequest{SessionID: "root-fanin", ProjectPath: "/p"})
	children := make([]SessionID, defaultMaxChildren)
	for i := range children {
		children[i] = mustOpen(t, s, created.Ref, "/p").SessionID
	}
	const collectors = 4
	completeErrs := make([]error, len(children))
	collectErrs := make([]error, collectors)
	violations := make([][]string, collectors)
	var completers sync.WaitGroup
	completers.Add(len(children))
	concurrently(len(children)+collectors, func(i int) {
		if i < len(children) {
			defer completers.Done()
			_, completeErrs[i] = s.Complete(ctx, CompleteRequest{SessionID: children[i], Content: "result of child " + strconv.Itoa(i), Summary: "done " + strconv.Itoa(i)})
			return
		}
		c := i - len(children)
		finished := make(chan struct{})
		go func() {
			completers.Wait()
			close(finished)
		}()
		done := map[SessionID]bool{}
		for last := false; !last; {
			select {
			case <-finished:
				last = true
			default:
			}
			// One collector long-polls, so waiters are woken while completions land.
			res, err := s.Collect(ctx, CollectRequest{Handoff: created.Ref, WaitSeconds: c % 2})
			if err != nil {
				collectErrs[c] = err
				return
			}
			if len(res.Children) != len(children) {
				violations[c] = append(violations[c], "collect returned "+strconv.Itoa(len(res.Children))+" children")
			}
			for _, ch := range res.Children {
				switch {
				case ch.Status == StatusDone && ch.ResultRef == "":
					violations[c] = append(violations[c], string(ch.SessionID)+" done without a result")
				case ch.Status == StatusOpen && done[ch.SessionID]:
					violations[c] = append(violations[c], string(ch.SessionID)+" went from done back to open")
				case ch.Status != StatusOpen && ch.Status != StatusDone:
					violations[c] = append(violations[c], string(ch.SessionID)+" has status "+string(ch.Status))
				}
				done[ch.SessionID] = done[ch.SessionID] || ch.Status == StatusDone
			}
		}
	})
	for i, err := range completeErrs {
		require.NoError(t, err, "complete %s", children[i])
	}
	for c := range collectors {
		require.NoError(t, collectErrs[c], "collector %d", c)
		assert.Empty(t, violations[c], "collector %d", c)
	}
	res, err := s.Collect(ctx, CollectRequest{Handoff: created.Ref})
	require.NoError(t, err)
	require.Len(t, res.Children, len(children))
	refs := map[string]bool{}
	for _, ch := range res.Children {
		assert.Equal(t, StatusDone, ch.Status, string(ch.SessionID))
		refs[ch.ResultRef] = true
	}
	assert.Len(t, refs, len(children), "each child has its own result")
}

// TestConcurrentTreeActivity is NFR-4 as one burst: sixteen children open one handoff and each
// posts, claims the same key, and searches at once, with no lost writes or duplicate grants.
func TestConcurrentTreeActivity(t *testing.T) {
	s := newTestService(t)
	ctx := context.Background()
	parent := SessionID("root-burst")
	trail.Record(trail.Entry{SessionID: string(parent), Tool: "search_semantic", Query: "retry backoff", ProjectPath: "/p", HitCount: 2})
	created := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: "/p", Mode: ModeFork})
	repeat := SearchEventFor(trail.Entry{Tool: "search_semantic", Query: "Retry  Backoff", QueryNorm: trail.NormalizeQuery("Retry  Backoff"), ProjectPath: "/p", HitCount: 2})
	sids := make([]SessionID, defaultMaxChildren)
	claims := make([]*ClaimResponse, defaultMaxChildren)
	annotations := make([]map[string]any, defaultMaxChildren)
	failures := make([]error, defaultMaxChildren)
	concurrently(defaultMaxChildren, func(i int) {
		opened, err := s.Open(ctx, OpenRequest{Handoff: created.Ref})
		if err != nil {
			failures[i] = err
			return
		}
		sids[i] = opened.SessionID
		if _, err := s.Post(ctx, PostRequest{SessionID: sids[i], Type: EntryTypeFinding, Text: "child " + strconv.Itoa(i) + " checked the retry path"}); err != nil {
			failures[i] = err
			return
		}
		if claims[i], err = s.Claim(ctx, ClaimRequest{SessionID: sids[i], Key: "retry.go"}); err != nil {
			failures[i] = err
			return
		}
		annotations[i] = s.Annotate(sids[i], repeat, nil)
		s.Touch(sids[i])
	})
	holders, positions := 0, map[int]bool{}
	for i := range sids {
		require.NoError(t, failures[i], "child %d", i)
		assert.Contains(t, annotations[i], "parent_trail_match", "child %s", sids[i])
		switch claims[i].Outcome {
		case ClaimGranted:
			holders++
		case ClaimQueued:
			assert.False(t, positions[claims[i].Position], "position %d handed out twice", claims[i].Position)
			positions[claims[i].Position] = true
		default:
			t.Fatalf("unexpected outcome %s", claims[i].Outcome)
		}
	}
	assert.Equal(t, 1, holders, "exactly one holder")
	assert.Len(t, positions, defaultMaxChildren-1)
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT COUNT(*) FROM handoff_children WHERE handoff_ref = ?`, created.Ref))
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT COUNT(*) FROM scratchpad_entries WHERE tree_id = ? AND type = 'finding'`, created.TreeID))
	assert.Equal(t, defaultMaxChildren-1, count(t, `SELECT COUNT(*) FROM handoff_claim_queue WHERE tree_id = ?`, created.TreeID))
	assert.Equal(t, 1, count(t, `SELECT COUNT(*) FROM handoff_claims WHERE tree_id = ?`, created.TreeID))
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT SUM(search_calls) FROM handoff_children WHERE handoff_ref = ?`, created.Ref), "every search counted")
	assert.Equal(t, defaultMaxChildren, count(t, `SELECT SUM(repeat_calls) FROM handoff_children WHERE handoff_ref = ?`, created.Ref), "every search is a parent repeat")
}
