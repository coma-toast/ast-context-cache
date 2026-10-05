package handoff

import (
	"context"
	"database/sql"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func claim(t *testing.T, s *realService, sid SessionID, key string) *ClaimResponse {
	t.Helper()
	res, err := s.Claim(context.Background(), ClaimRequest{SessionID: sid, Key: key})
	require.NoError(t, err)
	return res
}

func releaseAll(t *testing.T, s *realService, tree TreeID, sid SessionID) []string {
	t.Helper()
	var keys []string
	require.NoError(t, db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		keys, err = s.releaseAllTx(tx, tree, sid)
		return err
	}))
	return keys
}

func queueOrder(t *testing.T, tree TreeID, key string) []SessionID {
	t.Helper()
	rows, err := db.ContextDB.Query(`SELECT session_id FROM handoff_claim_queue WHERE tree_id = ? AND key = ? ORDER BY id`, tree, key)
	require.NoError(t, err)
	defer rows.Close()
	var out []SessionID
	for rows.Next() {
		var sid SessionID
		require.NoError(t, rows.Scan(&sid))
		out = append(out, sid)
	}
	return out
}

func setClaimWaitObserver(t *testing.T) *[]time.Duration {
	t.Helper()
	var waits []time.Duration
	prev := onClaimWait
	onClaimWait = func(d time.Duration) { waits = append(waits, d) }
	t.Cleanup(func() { onClaimWait = prev })
	return &waits
}

func TestNormalizeClaimKey(t *testing.T) {
	t.Parallel()
	tests := []struct{ in, want string }{
		{"x.go", "x.go"},
		{"  ./internal/../x.go ", "x.go"},
		{"internal//mcp/server.go", "internal/mcp/server.go"},
		{"internal/mcp/", "internal/mcp"},
		{"internal/a.go|Run|12", "internal/a.go|Run|12"},
		{"./a.go#Run@12", "./a.go#Run@12"},
		{"pkg.Func", "pkg.Func"},
		{"db migration", "db migration"},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			t.Parallel()
			got, err := normalizeClaimKey(tc.in)
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
	_, err := normalizeClaimKey(" ")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	_, err = normalizeClaimKey(string(make([]byte, maxClaimKeyBytes+1)))
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

// TestClaimQueueAndAutoGrant covers AC18: A holds x.go, B and C queue in that order, and when A
// completes B is granted, C moves to position 1, and B is told once.
func TestClaimQueueAndAutoGrant(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	waits := setClaimWaitObserver(t)
	tt := seedBareTree(t, "root-claim", "alpha", "beta", "gamma")
	a, b, c := tt.children[0], tt.children[1], tt.children[2]
	assert.Equal(t, &ClaimResponse{Key: "x.go", Outcome: ClaimGranted, Holder: a}, claim(t, s, a, "./x.go"))
	assert.Equal(t, &ClaimResponse{Key: "x.go", Outcome: ClaimHeld, Holder: a}, claim(t, s, a, "x.go"))
	want := &ClaimResponse{Key: "x.go", Outcome: ClaimQueued, Holder: a, HolderLabel: "alpha", Position: 1}
	res, err := s.Claim(ctx, ClaimRequest{SessionID: b, Key: "x.go", Reason: "fix the retry"})
	require.NoError(t, err)
	assert.Equal(t, want, res)
	want.Position = 2
	assert.Equal(t, want, claim(t, s, c, "x.go"))
	assert.Equal(t, want, claim(t, s, c, "x.go"), "claiming again reports the same place in line")
	assert.Equal(t, []SessionID{b, c}, queueOrder(t, tt.tree, "x.go"))

	// CL-8: claims and queues show up in scratchpad reads.
	got, err := s.Read(ctx, ReadRequest{SessionID: tt.root})
	require.NoError(t, err)
	require.Len(t, got.Claims, 1)
	view := got.Claims[0]
	assert.Equal(t, "x.go", view.Key)
	assert.Equal(t, a, view.Holder)
	assert.Equal(t, "alpha", view.HolderLabel)
	require.Len(t, view.Queue, 2)
	assert.Equal(t, QueuedClaim{SessionID: b, Reason: "fix the retry", EnqueuedAt: view.Queue[0].EnqueuedAt, Position: 1}, view.Queue[0])
	assert.Equal(t, c, view.Queue[1].SessionID)
	assert.Equal(t, 2, view.Queue[1].Position)
	var claimTexts []string
	for _, e := range got.Entries {
		if e.Type == EntryTypeClaim {
			claimTexts = append(claimTexts, e.Text)
		}
	}
	assert.Equal(t, []string{"claimed x.go", "queued for x.go behind " + string(a), "queued for x.go behind " + string(a)}, claimTexts)

	grants, err := s.PendingGrants(b)
	require.NoError(t, err)
	assert.Empty(t, grants, "nothing granted from a queue yet")
	assert.Equal(t, []string{"x.go"}, releaseAll(t, s, tt.tree, a))
	assert.Equal(t, &ClaimResponse{Key: "x.go", Outcome: ClaimHeld, Holder: b}, claim(t, s, b, "x.go"))
	assert.Equal(t, &ClaimResponse{Key: "x.go", Outcome: ClaimQueued, Holder: b, HolderLabel: "beta", Position: 1}, claim(t, s, c, "x.go"))
	assert.Len(t, *waits, 1, "the grant's wait is observed")
	grants, err = s.PendingGrants(b)
	require.NoError(t, err)
	require.Len(t, grants, 1)
	assert.Equal(t, tt.tree, grants[0].TreeID)
	assert.Equal(t, "x.go", grants[0].Key)
	assert.NotEmpty(t, grants[0].GrantedAt)
	grants, err = s.PendingGrants(b)
	require.NoError(t, err)
	assert.Empty(t, grants, "a grant is reported once")
	grants, err = s.PendingGrants(c)
	require.NoError(t, err)
	assert.Empty(t, grants)
	grants, err = s.PendingGrants("loner")
	require.NoError(t, err)
	assert.Nil(t, grants)
	assert.Equal(t, 1, count(t, `SELECT COUNT(*) FROM scratchpad_entries WHERE tree_id = ? AND type = 'claim' AND author_session_id = ? AND text LIKE 'granted x.go after %'`, tt.tree, b))
}

func TestRelease(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	tt := seedBareTree(t, "root-release", "alpha", "beta", "gamma")
	a, b, c := tt.children[0], tt.children[1], tt.children[2]
	claim(t, s, a, "x.go")
	claim(t, s, b, "x.go")
	claim(t, s, c, "x.go")
	res, err := s.Release(ctx, ReleaseRequest{SessionID: c, Key: "x.go"})
	require.NoError(t, err)
	assert.Equal(t, &ReleaseResponse{Key: "x.go", Released: true}, res, "a waiter leaves the queue")
	res, err = s.Release(ctx, ReleaseRequest{SessionID: a, Key: "./x.go"})
	require.NoError(t, err)
	assert.Equal(t, &ReleaseResponse{Key: "x.go", Released: true, GrantedTo: b}, res)
	assert.Empty(t, queueOrder(t, tt.tree, "x.go"))
	res, err = s.Release(ctx, ReleaseRequest{SessionID: b, Key: "x.go"})
	require.NoError(t, err)
	assert.Equal(t, &ReleaseResponse{Key: "x.go", Released: true}, res, "nobody left to grant")
	assert.Equal(t, &ClaimResponse{Key: "x.go", Outcome: ClaimGranted, Holder: c}, claim(t, s, c, "x.go"))
	_, err = s.Release(ctx, ReleaseRequest{SessionID: a, Key: "x.go"})
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
	_, err = s.Release(ctx, ReleaseRequest{SessionID: "loner", Key: "x.go"})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound))
	_, err = s.Claim(ctx, ClaimRequest{SessionID: "loner", Key: "x.go"})
	assert.True(t, errs.HasCode(err, CodeHandoffNotFound))
	_, err = s.Claim(ctx, ClaimRequest{SessionID: a, Key: ""})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

// TestReleaseAllDropsHeldAndQueued covers completion and abandonment: every held claim goes to
// its next waiter and the session's own queued requests are withdrawn.
func TestReleaseAllDropsHeldAndQueued(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	tt := seedBareTree(t, "root-all", "alpha", "beta", "gamma")
	a, b, c := tt.children[0], tt.children[1], tt.children[2]
	claim(t, s, a, "x.go")
	claim(t, s, a, "y.go")
	claim(t, s, b, "z.go")
	claim(t, s, a, "z.go")
	claim(t, s, c, "y.go")
	assert.Equal(t, []string{"x.go", "y.go"}, releaseAll(t, s, tt.tree, a))
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_claims WHERE holder_session_id = ?`, a))
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_claim_queue WHERE session_id = ?`, a))
	assert.Equal(t, &ClaimResponse{Key: "y.go", Outcome: ClaimHeld, Holder: c}, claim(t, s, c, "y.go"))
	assert.Equal(t, &ClaimResponse{Key: "x.go", Outcome: ClaimGranted, Holder: b}, claim(t, s, b, "x.go"), "x.go had no waiter and is free")
	assert.Empty(t, releaseAll(t, s, tt.tree, a), "nothing left to release")

	// markAbandoned releases through the same path.
	exec(t, `UPDATE handoff_children SET last_activity_at = '2000-01-01 00:00:00' WHERE child_session_id = ?`, b)
	claim(t, s, c, "z.go")
	n, err := s.markAbandoned()
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	assert.Equal(t, &ClaimResponse{Key: "z.go", Outcome: ClaimHeld, Holder: c}, claim(t, s, c, "z.go"))
	grants, err := s.PendingGrants(c)
	require.NoError(t, err)
	assert.Len(t, grants, 2, "y.go and z.go")
}

// TestClaimDeadlockRisk covers AC19 and CL-6.
func TestClaimDeadlockRisk(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	ctx := context.Background()
	tt := seedBareTree(t, "root-deadlock", "alpha", "beta", "gamma")
	a, b, c := tt.children[0], tt.children[1], tt.children[2]
	claim(t, s, a, "x.go")
	claim(t, s, b, "y.go")
	assert.Equal(t, ClaimQueued, claim(t, s, a, "y.go").Outcome)
	_, err := s.Claim(ctx, ClaimRequest{SessionID: b, Key: "x.go"})
	require.True(t, errs.HasCode(err, CodeClaimDeadlockRisk), "%v", err)
	m := ErrorMap(err)
	assert.Equal(t, string(CodeClaimDeadlockRisk), m["error"])
	cycle := m["details"].(map[string]any)["cycle"]
	assert.Equal(t, string(b)+" waits for x.go held by "+string(a)+"; "+string(a)+" waits for y.go held by "+string(b), cycle)
	assert.Empty(t, queueOrder(t, tt.tree, "x.go"), "the request is not queued")

	// A longer cycle: c holds w.go and b waits for it; c then claiming x.go closes c→a→b→c.
	claim(t, s, c, "w.go")
	assert.Equal(t, ClaimQueued, claim(t, s, b, "w.go").Outcome)
	_, err = s.Claim(ctx, ClaimRequest{SessionID: c, Key: "x.go"})
	require.True(t, errs.HasCode(err, CodeClaimDeadlockRisk), "%v", err)
	assert.Contains(t, errs.FieldsOf(err)["cycle"], string(b)+" waits for w.go held by "+string(c))
	assert.Equal(t, ClaimQueued, claim(t, s, tt.root, "x.go").Outcome, "a waiter outside the cycle still queues")
}

// TestClaimConcurrentFIFO covers AC20 and NFR-4: 16 children claiming one key at once leave one
// holder and 15 waiters whose positions match their arrival order, and releases grant in that
// order without duplicates.
func TestClaimConcurrentFIFO(t *testing.T) {
	initHandoffDB(t)
	s := newService(nil)
	labels := make([]string, 16)
	for i := range labels {
		labels[i] = "child"
	}
	tt := seedBareTree(t, "root-race", labels...)
	results := make([]*ClaimResponse, len(tt.children))
	errsOut := make([]error, len(tt.children))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, sid := range tt.children {
		wg.Go(func() {
			<-start
			results[i], errsOut[i] = s.Claim(context.Background(), ClaimRequest{SessionID: sid, Key: "hot.go"})
		})
	}
	close(start)
	wg.Wait()
	var holder SessionID
	byPosition := map[int]SessionID{}
	for i, res := range results {
		require.NoError(t, errsOut[i])
		switch res.Outcome {
		case ClaimGranted:
			require.Empty(t, holder, "exactly one holder")
			holder = tt.children[i]
		case ClaimQueued:
			_, dup := byPosition[res.Position]
			require.False(t, dup, "position %d handed out twice", res.Position)
			byPosition[res.Position] = tt.children[i]
		default:
			t.Fatalf("unexpected outcome %s", res.Outcome)
		}
	}
	require.NotEmpty(t, holder)
	require.Len(t, byPosition, 15)
	order := queueOrder(t, tt.tree, "hot.go")
	require.Len(t, order, 15)
	for pos, sid := range order {
		assert.Equal(t, byPosition[pos+1], sid, "position %d matches arrival order", pos+1)
	}
	for _, next := range order {
		res, err := s.Release(context.Background(), ReleaseRequest{SessionID: holder, Key: "hot.go"})
		require.NoError(t, err)
		require.Equal(t, next, res.GrantedTo, "grants follow FIFO order")
		holder = next
	}
	assert.Equal(t, 1, count(t, `SELECT COUNT(*) FROM handoff_claims WHERE tree_id = ?`, tt.tree))
	assert.Equal(t, 15, count(t, `SELECT COUNT(DISTINCT session_id) FROM handoff_claim_grants WHERE tree_id = ?`, tt.tree))
	assert.Equal(t, 15, count(t, `SELECT COUNT(*) FROM handoff_claim_grants WHERE tree_id = ?`, tt.tree), "no duplicate grants")
	assert.True(t, slices.Contains(tt.children, holder))
}
