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
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

// AC2 and HO-3: what the parent does after creating a handoff doesn't reach its snapshot.
func TestOpenSnapshotImmutable(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	parent := SessionID("parent-ac2")
	recordSearches(parent, project, 10, 2)
	note, err := contextnotes.Store(string(parent), "design notes", "design", project, nil, "", nil, nil)
	require.NoError(t, err)
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project, CtxRefs: []string{note.Ref}, Pointers: []PointerInput{{Key: "svc.go|Alpha", Note: "start here"}}})
	_, err = contextnotes.Store(string(parent), "later notes", "later", project, nil, "", nil, nil)
	require.NoError(t, err)
	recordSearches(parent, project, 15, 2)
	_, err = memory.Store(memory.StoreInput{Kind: memory.KindProcedure, SessionID: string(parent), Rule: "a later rule"})
	require.NoError(t, err)

	o := mustOpen(t, s, resp.Ref, "")
	assert.Equal(t, ChildSessionID(resp.Ref, 1), o.SessionID)
	assert.Equal(t, resp.TreeID, o.TreeID)
	assert.Equal(t, ModeFresh, o.Mode)
	assert.Equal(t, "investigate the retry path", o.Brief)
	assert.Len(t, o.Trail, 10, "the original 10 searches only")
	assert.Equal(t, "query 9", o.Trail[0].Query, "newest first")
	require.Len(t, o.Notes, 1, "no new note")
	assert.Equal(t, note.Ref, o.Notes[0].Ref)
	assert.Empty(t, o.Memory, "memory stored after the snapshot isn't in it")
	require.Len(t, o.Pointers, 1)
	assert.Equal(t, PointerDigest{ID: o.Pointers[0].ID, Key: "svc.go|Alpha", Note: "start here", Kind: "function"}, o.Pointers[0])
	assert.False(t, o.Truncated)
	assert.Nil(t, o.Scratchpad, "an empty scratchpad is left out")
	assert.True(t, s.IsTreeSession(o.SessionID))
	assert.Positive(t, o.TokensAvailable)
	assert.Equal(t, o.TokensUsed, o.TokensDelivered)
	assert.Equal(t, o.TokensAvailable, count(t, `SELECT tokens_available FROM handoff_children WHERE child_session_id = ?`, o.SessionID))
	assert.Equal(t, o.TokensUsed, count(t, `SELECT tokens_delivered FROM handoff_children WHERE child_session_id = ?`, o.SessionID))
}

// AC4 and OP-5, OP-11: a fork child starts deduped against the parent's manifest, and every
// child gets the snapshot memory as its own session entries.
func TestOpenForkSeedsDedupAndClonesMemory(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	parent := SessionID("parent-fork")
	returnSymbols(t, parent, project, "Alpha", "Beta")
	mem, err := memory.Store(memory.StoreInput{Kind: memory.KindProcedure, SessionID: string(parent), Rule: "retry with jitter"})
	require.NoError(t, err)
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project, Mode: ModeFork})
	o := mustOpen(t, s, resp.Ref, "")
	assert.Equal(t, ModeFork, o.Mode)
	assert.Equal(t, astcontext.ReturnedKeys(string(parent)), astcontext.ReturnedKeys(string(o.SessionID)))
	results, savings := packedFor(t, o.SessionID, project, "Alpha")
	assert.Empty(t, results, "the parent's symbol is deduped")
	assert.Equal(t, 1, savings.DedupedCount)
	assert.Positive(t, savings.DedupTokensSaved)
	cloned, err := memory.ActiveForSession(string(o.SessionID))
	require.NoError(t, err)
	require.Len(t, cloned, 1)
	assert.Equal(t, "retry with jitter", cloned[0].Rule)
	assert.Equal(t, mem.Ref, cloned[0].SourceRef)

	fresh := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project})
	fo := mustOpen(t, s, fresh.Ref, "")
	assert.Empty(t, astcontext.ReturnedKeys(string(fo.SessionID)), "a fresh child starts with empty dedup")
	results, _ = packedFor(t, fo.SessionID, project, "Alpha")
	assert.Len(t, results, 1)
}

// AC14 and HO-9, OP-2: opens are capped per handoff, resumes are not, and a resume revives an
// abandoned child.
func TestOpenChildrenCapAndResume(t *testing.T) {
	s := newTestService(t)
	require.NoError(t, db.SetSetting(SettingMaxChildren, "2"))
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-cap", ProjectPath: "/p"})
	first := mustOpen(t, s, resp.Ref, "")
	mustOpen(t, s, resp.Ref, "")
	_, err := s.Open(context.Background(), OpenRequest{Handoff: resp.Ref})
	require.True(t, errs.HasCode(err, CodeHandoffChildrenExceeded), "%v", err)
	assert.Equal(t, string(CodeHandoffChildrenExceeded), ErrorMap(err)["error"])

	exec(t, `UPDATE handoff_children SET status = 'abandoned' WHERE child_session_id = ?`, first.SessionID)
	r, err := s.Open(context.Background(), OpenRequest{Handoff: resp.Ref, SessionID: first.SessionID, ProjectPath: "/elsewhere"})
	require.NoError(t, err)
	assert.True(t, r.Resumed)
	assert.Equal(t, first.SessionID, r.SessionID)
	assert.Equal(t, StatusOpen, childStatus(t, first.SessionID))
	assert.Equal(t, 2, count(t, `SELECT child_count FROM handoffs WHERE ref = ?`, resp.Ref), "no count bump")
	assert.Equal(t, 2, count(t, `SELECT COUNT(*) FROM handoff_children`))
	var project string
	require.NoError(t, db.ContextDB.QueryRow(`SELECT project_path FROM handoff_children WHERE child_session_id = ?`, first.SessionID).Scan(&project))
	assert.Equal(t, "/elsewhere", project)

	_, err = s.Open(context.Background(), OpenRequest{Handoff: resp.Ref, SessionID: "stranger"})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "%v", err)
	other := mustCreate(t, s, CreateRequest{SessionID: "parent-cap", ProjectPath: "/p"})
	_, err = s.Open(context.Background(), OpenRequest{Handoff: other.Ref, SessionID: first.SessionID})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "a child of another handoff: %v", err)
}

func TestOpenErrors(t *testing.T) {
	s := newTestService(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-exp"})
	_, err := s.Open(context.Background(), OpenRequest{Handoff: "hof_0000000000000000"})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound), "%v", err)
	_, err = s.Open(context.Background(), OpenRequest{Handoff: "nope"})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
	_, err = s.Open(context.Background(), OpenRequest{Handoff: resp.Ref, Next: &PageCursor{Section: SectionTrail}})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "paging needs the child id: %v", err)

	setClock(t, now.Add(7*24*time.Hour+time.Minute))
	_, err = s.Open(context.Background(), OpenRequest{Handoff: resp.Ref})
	assert.True(t, errs.HasCode(err, CodeHandoffExpired), "%v", err)
	assert.Equal(t, string(CodeHandoffExpired), ErrorMap(err)["error"])
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_children`))
	_, err = s.Create(context.Background(), CreateRequest{SessionID: "parent-exp", Brief: "again"})
	require.NoError(t, err)
	assert.Equal(t, 2, count(t, `SELECT COUNT(*) FROM handoff_trees`), "an expired tree isn't reused")
}

// OP-3: the default digest stays within its budget, and the cursor pages through the rest.
func TestOpenDigestBudgetAndPaging(t *testing.T) {
	s := newTestService(t)
	parent := SessionID("parent-page")
	recordSearches(parent, "/p", 120, 4)
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: "/p"})
	o, err := s.Open(context.Background(), OpenRequest{Handoff: resp.Ref})
	require.NoError(t, err)
	assert.LessOrEqual(t, o.TokensUsed, 1500, "the default budget")
	assert.InDelta(t, responseTokens(o), o.TokensUsed, 1)
	require.True(t, o.Truncated)
	require.NotNil(t, o.Next)
	assert.Equal(t, SectionTrail, o.Next.Section)
	assert.Equal(t, len(o.Trail), o.Next.Offset)

	seen := map[int64]bool{}
	for _, tr := range o.Trail {
		seen[tr.ID] = true
	}
	next := o.Next
	for next != nil {
		page, err := s.Open(context.Background(), OpenRequest{Handoff: resp.Ref, SessionID: o.SessionID, TokenBudget: 400, Next: next})
		require.NoError(t, err)
		assert.LessOrEqual(t, page.TokensUsed, 400)
		assert.Empty(t, page.Brief, "continuation pages skip the brief")
		require.NotEmpty(t, page.Trail)
		for _, tr := range page.Trail {
			assert.False(t, seen[tr.ID], "no item repeats across pages")
			seen[tr.ID] = true
		}
		next = page.Next
	}
	assert.Len(t, seen, 120)
	assert.Equal(t, 1, count(t, `SELECT child_count FROM handoffs WHERE ref = ?`, resp.Ref), "paging doesn't mint children")
}

// OP-3, SP-7, CL-8: the digest summarizes the tree's scratchpad.
func TestOpenScratchpadDigest(t *testing.T) {
	s := newTestService(t)
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-sp"})
	o := mustOpen(t, s, resp.Ref, "")
	for _, e := range []struct{ typ, text, refs string }{
		{"finding", "found the retry loop in client.go", ""},
		{"dead_end", "server.go has no retries", ""},
		{"trail", "search_semantic: backoff (0 hits)", `{"match_key":"k","hit_count":0,"zero_hit":true}`},
		{"trail", "search_semantic: retry (4 hits)", `{"match_key":"k2","hit_count":4,"zero_hit":false}`},
		{"finding", "retracted", ""},
	} {
		exec(t, `INSERT INTO scratchpad_entries (tree_id, author_session_id, type, text, refs_json) VALUES (?, ?, ?, ?, ?)`,
			resp.TreeID, o.SessionID, e.typ, e.text, nullIfEmpty(e.refs))
	}
	exec(t, `UPDATE scratchpad_entries SET retracted_at = datetime('now') WHERE text = 'retracted'`)
	exec(t, `INSERT INTO handoff_claims (tree_id, key, holder_session_id, reason) VALUES (?, 'client.go', ?, 'editing')`, resp.TreeID, o.SessionID)
	exec(t, `INSERT INTO handoff_claim_queue (tree_id, key, session_id) VALUES (?, 'client.go', 'waiter')`, resp.TreeID)

	r, err := s.Open(context.Background(), OpenRequest{Handoff: resp.Ref, SessionID: o.SessionID})
	require.NoError(t, err)
	require.NotNil(t, r.Scratchpad)
	assert.Equal(t, map[EntryType]int{EntryTypeFinding: 1, EntryTypeDeadEnd: 1, EntryTypeTrail: 2}, r.Scratchpad.Counts)
	var latest, deadEnds []string
	for _, e := range r.Scratchpad.Latest {
		latest = append(latest, e.Headline)
	}
	for _, e := range r.Scratchpad.DeadEnds {
		deadEnds = append(deadEnds, e.Headline)
	}
	assert.Equal(t, []string{"server.go has no retries", "found the retry loop in client.go"}, latest, "newest first, no trail entries")
	assert.Equal(t, []string{"search_semantic: backoff (0 hits)", "server.go has no retries"}, deadEnds)
	require.Len(t, r.Scratchpad.Claims, 1)
	c := r.Scratchpad.Claims[0]
	assert.Equal(t, "client.go", c.Key)
	assert.Equal(t, o.SessionID, c.Holder)
	assert.Equal(t, "editing", c.Reason)
	require.Len(t, c.Queue, 1)
	assert.Equal(t, QueuedClaim{SessionID: "waiter", EnqueuedAt: c.Queue[0].EnqueuedAt, Position: 1}, c.Queue[0])
}
