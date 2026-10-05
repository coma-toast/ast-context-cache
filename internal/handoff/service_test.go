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
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

// DB-backed tests don't call t.Parallel: the db pools and nowFunc are package globals.

// testTree is a seeded tree: a root session, one handoff, and its children.
type testTree struct {
	tree     TreeID
	root     SessionID
	ref      HandoffRef
	children []SessionID
}

func setClock(t *testing.T, now time.Time) {
	t.Helper()
	prev := nowFunc
	nowFunc = func() time.Time { return now }
	t.Cleanup(func() { nowFunc = prev })
}

func exec(t *testing.T, q string, args ...any) {
	t.Helper()
	_, err := db.ContextDB.Exec(q, args...)
	require.NoError(t, err, q)
}

func count(t *testing.T, q string, args ...any) int {
	t.Helper()
	var n int
	require.NoError(t, db.ContextDB.QueryRow(q, args...).Scan(&n), q)
	return n
}

// seedTree inserts a tree with one row in every handoff table, accessed and active at at.
func seedTree(t *testing.T, root SessionID, nChildren int, at time.Time) testTree {
	t.Helper()
	tree, err := NewTreeID()
	require.NoError(t, err)
	ref, err := NewHandoffRef()
	require.NoError(t, err)
	ts := sqlTime(at)
	tt := testTree{tree: tree, root: root, ref: ref}
	exec(t, `INSERT INTO handoff_trees (tree_id, root_session_id, project_path, created_at, last_access_at) VALUES (?, ?, '/p', ?, ?)`, tree, root, ts, ts)
	exec(t, `INSERT INTO handoffs (ref, tree_id, parent_session_id, depth, mode, brief, project_path, child_count, created_at, last_access_at)
		VALUES (?, ?, ?, 1, 'fork', 'do it', '/p', ?, ?, ?)`, ref, tree, root, nChildren, ts, ts)
	exec(t, `INSERT INTO handoff_snapshot_items (handoff_ref, section, ord, item_key) VALUES (?, 'pointer', 0, 'a.go|A|1')`, ref)
	for i := range nChildren {
		child := ChildSessionID(ref, i+1)
		tt.children = append(tt.children, child)
		exec(t, `INSERT INTO handoff_children (child_session_id, handoff_ref, tree_id, status, project_path, opened_at, last_activity_at)
			VALUES (?, ?, ?, 'open', '/p', ?, ?)`, child, ref, tree, ts, ts)
		exec(t, `INSERT INTO handoff_results (child_session_id, result_ref, created_at) VALUES (?, 'ctx_result', ?)`, child, ts)
		exec(t, `INSERT INTO scratchpad_entries (tree_id, author_session_id, type, text) VALUES (?, ?, 'finding', 'found it')`, tree, child)
		exec(t, `INSERT INTO handoff_claim_grants (session_id, tree_id, key) VALUES (?, ?, ?)`, child, tree, "g"+string(child))
	}
	exec(t, `INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES (?, 'a.go', ?)`, tree, root)
	exec(t, `INSERT INTO handoff_claim_queue (tree_id, key, session_id) VALUES (?, 'a.go', 'waiter')`, tree)
	return tt
}

func treeRowCount(t *testing.T, tt testTree) int {
	t.Helper()
	return count(t, `SELECT
		(SELECT COUNT(*) FROM handoff_trees WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoffs WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_snapshot_items WHERE handoff_ref = ?2) +
		(SELECT COUNT(*) FROM handoff_children WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_results WHERE child_session_id LIKE ?2 || '.c%') +
		(SELECT COUNT(*) FROM scratchpad_entries WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_claims WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_claim_queue WHERE tree_id = ?1) +
		(SELECT COUNT(*) FROM handoff_claim_grants WHERE tree_id = ?1)`, tt.tree, tt.ref)
}

func childStatus(t *testing.T, sid SessionID) Status {
	t.Helper()
	var s Status
	require.NoError(t, db.ContextDB.QueryRow(`SELECT status FROM handoff_children WHERE child_session_id = ?`, sid).Scan(&s))
	return s
}

func TestMarkAbandoned(t *testing.T) {
	dbtest.Init(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	s := newService(nil)
	stale := seedTree(t, "root-stale", 2, now.Add(-31*time.Minute))
	fresh := seedTree(t, "root-fresh", 1, now.Add(-29*time.Minute))
	exec(t, `UPDATE handoff_children SET status = 'done' WHERE child_session_id = ?`, stale.children[1])
	wake := s.waiters.wait(stale.tree)
	n, err := s.markAbandoned()
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	assert.Equal(t, StatusAbandoned, childStatus(t, stale.children[0]))
	assert.Equal(t, StatusDone, childStatus(t, stale.children[1]), "completed children stay completed")
	assert.Equal(t, StatusOpen, childStatus(t, fresh.children[0]), "inside the 30-minute window")
	select {
	case <-wake:
	default:
		t.Fatal("collect waiters on the tree are woken")
	}
	n, err = s.markAbandoned()
	require.NoError(t, err)
	assert.Zero(t, n, "already abandoned")

	require.NoError(t, db.SetSetting(SettingChildInactiveMinutes, "10"))
	n, err = s.markAbandoned()
	require.NoError(t, err)
	assert.Equal(t, 1, n, "the window comes from settings")
	assert.Equal(t, StatusAbandoned, childStatus(t, fresh.children[0]))
}

func TestFlushTree(t *testing.T) {
	dbtest.Init(t)
	s := newService(nil)
	now := time.Now()
	tt := seedTree(t, "root-flush", 2, now)
	other := seedTree(t, "root-other", 1, now)
	before := treeRowCount(t, other)
	child := tt.children[0]
	note, err := contextnotes.Store(string(child), "child working notes", "work", "/p", nil, "", nil, nil)
	require.NoError(t, err)
	result, err := contextnotes.Store(string(child), "the result", "result", "/p", nil, contextnotes.KindHandoffResult, nil, nil)
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{Kind: memory.KindProcedure, SessionID: string(child), Rule: "cloned parent rule"})
	require.NoError(t, err)
	promoted, err := memory.Store(memory.StoreInput{Kind: memory.KindProcedure, SessionID: string(tt.root), Rule: "promoted rule", SourceRef: result.Ref})
	require.NoError(t, err)
	parentNote, err := contextnotes.Store(string(tt.root), "parent notes", "", "/p", nil, "", nil, nil)
	require.NoError(t, err)
	astcontext.MarkReturned(string(child), astcontext.ReturnedSymbol{File: "/p/a.go", Name: "A", StartLine: 1})
	astcontext.MarkReturned(string(tt.root), astcontext.ReturnedSymbol{File: "/p/b.go", Name: "B", StartLine: 2})
	require.True(t, s.IsTreeSession(child))
	require.True(t, s.IsTreeSession(tt.root))
	wake := s.waiters.wait(tt.tree)

	res, err := s.flushTree(tt.tree)
	require.NoError(t, err)
	assert.Equal(t, &FlushResponse{TreeID: tt.tree, Handoffs: 1, Children: 2, NotesDeleted: 2, MemoryDeleted: 1}, res)
	assert.Zero(t, treeRowCount(t, tt), "every handoff table row of the tree")
	assert.Equal(t, before, treeRowCount(t, other), "other trees untouched")
	for _, ref := range []string{note.Ref, result.Ref} {
		_, err := contextnotes.Peek(ref)
		assert.True(t, errs.HasCode(err, errs.CodeNotFound), "child note %s deleted", ref)
	}
	_, err = contextnotes.Peek(parentNote.Ref)
	assert.NoError(t, err, "the parent's own notes stay")
	childMem, err := memory.ActiveForSession(string(child))
	require.NoError(t, err)
	assert.Empty(t, childMem)
	parentMem, err := memory.ActiveForSession(string(tt.root))
	require.NoError(t, err)
	require.Len(t, parentMem, 1, "promoted memory outlives the tree")
	assert.Equal(t, promoted.Ref, parentMem[0].Ref)
	assert.Empty(t, astcontext.ReturnedKeys(string(child)), "child dedup rows expire with the tree")
	assert.Contains(t, astcontext.ReturnedKeys(string(tt.root)), "/p/b.go|B|2", "the parent's dedup stays")
	db.FlushWriteBuffers()
	var dedup int
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM sessions WHERE session_id = ?`, string(child)).Scan(&dedup))
	assert.Zero(t, dedup)
	assert.False(t, s.IsTreeSession(child))
	assert.False(t, s.IsTreeSession(tt.root))
	select {
	case <-wake:
	default:
		t.Fatal("collect waiters on the tree are woken")
	}

	res, err = s.flushTree(tt.tree)
	require.NoError(t, err, "flushing a gone tree is a no-op")
	assert.Equal(t, &FlushResponse{TreeID: tt.tree}, res)
}

func TestSweepExpiresTreesPastTTL(t *testing.T) {
	dbtest.Init(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	s := newService(nil)
	expired := seedTree(t, "root-old", 1, now.Add(-7*24*time.Hour-time.Minute))
	live := seedTree(t, "root-new", 1, now.Add(-7*24*time.Hour+time.Minute))
	n, err := s.sweep()
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	assert.Zero(t, treeRowCount(t, expired))
	assert.NotZero(t, treeRowCount(t, live))
	require.NoError(t, db.SetSetting(SettingTTLDays, "1"))
	n, err = s.sweep()
	require.NoError(t, err)
	assert.Equal(t, 1, n, "the TTL comes from settings")
	assert.Zero(t, treeRowCount(t, live))
}

func TestTreeIndex(t *testing.T) {
	dbtest.Init(t)
	s := newService(nil)
	tt := seedTree(t, "root-idx", 1, time.Now())
	e, ok := s.trees.lookup(tt.children[0])
	require.True(t, ok)
	assert.Equal(t, treeEntry{tree: tt.tree, handoff: tt.ref, mode: ModeFork, isChild: true}, e)
	e, ok = s.trees.lookup(tt.root)
	require.True(t, ok)
	assert.Equal(t, treeEntry{tree: tt.tree}, e)
	assert.False(t, s.IsTreeSession("loner"))
	assert.False(t, s.IsTreeSession(""))

	// A remembered miss is answered from memory; put is how a session joins a tree.
	exec(t, `INSERT INTO handoff_trees (tree_id, root_session_id) VALUES ('hft_late', 'loner')`)
	assert.False(t, s.IsTreeSession("loner"), "no database read for a known non-tree session")
	s.trees.put("loner", treeEntry{tree: "hft_late"})
	assert.True(t, s.IsTreeSession("loner"))
	s.trees.forgetTree("hft_late")
	assert.True(t, s.IsTreeSession("loner"), "rehydrates from handoff_trees")

	for i := range maxNegativeEntries + 10 {
		s.trees.lookup(SessionID("miss-" + time.Duration(i).String()))
	}
	assert.LessOrEqual(t, s.trees.negatives, maxNegativeEntries)
	assert.True(t, s.IsTreeSession(tt.root), "positive entries survive the negative cap")
}

func TestUnimplementedMethods(t *testing.T) {
	dbtest.Init(t)
	s := newService(nil)
	ctx := context.Background()
	calls := map[string]func() error{
		"create":   func() error { _, err := s.Create(ctx, CreateRequest{}); return err },
		"open":     func() error { _, err := s.Open(ctx, OpenRequest{}); return err },
		"expand":   func() error { _, err := s.Expand(ctx, ExpandRequest{}); return err },
		"complete": func() error { _, err := s.Complete(ctx, CompleteRequest{}); return err },
		"collect":  func() error { _, err := s.Collect(ctx, CollectRequest{}); return err },
		"list":     func() error { _, err := s.List(ctx, ListRequest{}); return err },
		"status":   func() error { _, err := s.Status(ctx, StatusRequest{}); return err },
		"flush":    func() error { _, err := s.Flush(ctx, FlushRequest{}); return err },
		"post":     func() error { _, err := s.Post(ctx, PostRequest{}); return err },
		"read":     func() error { _, err := s.Read(ctx, ReadRequest{}); return err },
		"retract":  func() error { _, err := s.Retract(ctx, RetractRequest{}); return err },
		"claim":    func() error { _, err := s.Claim(ctx, ClaimRequest{}); return err },
		"release":  func() error { _, err := s.Release(ctx, ReleaseRequest{}); return err },
		"grants":   func() error { _, err := s.PendingGrants("x"); return err },
	}
	for name, call := range calls {
		t.Run(name, func(t *testing.T) {
			err := call()
			assert.True(t, errs.HasCode(err, errs.CodeUnsupported), "%v", err)
			assert.Equal(t, "unsupported", ErrorMap(err)["error"])
		})
	}
	assert.Nil(t, s.Annotate("x", SearchEvent{}, nil))
	s.Touch("x")
}

func TestStartInstallsDefault(t *testing.T) {
	dbtest.Init(t)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	t.Cleanup(func() { SetDefault(nil) })
	assert.Nil(t, Default())
	svc := Start(ctx, nil)
	assert.Same(t, svc.(*realService), Default().(*realService))
	SetDefault(nil)
	assert.Nil(t, Default())
}
