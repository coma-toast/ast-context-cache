package handoff

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestTreeViews(t *testing.T) {
	dbtest.Init(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	older := seedTree(t, "root-older", 1, now.Add(-8*24*time.Hour))
	newer := seedTree(t, "root-newer", 2, now.Add(-time.Hour))
	exec(t, `UPDATE handoff_trees SET created_at = ? WHERE tree_id = ?`, sqlTime(now.Add(-8*24*time.Hour)), older.tree)
	c1, c2 := newer.children[0], newer.children[1]
	exec(t, `UPDATE handoff_children SET search_calls = 4, repeat_calls = 1, tokens_available = 900, tokens_delivered = 300,
		status = 'done', result_ref = 'ctx_r', summary = 'did it' WHERE child_session_id = ?`, c1)
	exec(t, `UPDATE handoff_children SET search_calls = 6, repeat_calls = 4, tokens_available = 100, tokens_delivered = 250
		WHERE child_session_id = ?`, c2)
	exec(t, `INSERT INTO handoff_claims (tree_id, key, holder_session_id) VALUES (?, 'b.go', ?)`, newer.tree, c2)
	exec(t, `INSERT INTO handoff_claim_queue (tree_id, key, session_id) VALUES (?, 'b.go', ?)`, newer.tree, c1)

	trees, err := TreeViews(0)
	require.NoError(t, err)
	require.Len(t, trees, 2)
	assert.Equal(t, newer.tree, trees[0].TreeID, "newest first")
	assert.Equal(t, older.tree, trees[1].TreeID)
	assert.True(t, trees[1].Expired, "past the 7-day TTL")
	tv := trees[0]
	assert.False(t, tv.Expired)
	assert.Equal(t, sqlTime(now.Add(-time.Hour).Add(7*24*time.Hour)), tv.ExpiresAt)
	assert.Equal(t, SessionID("root-newer"), tv.RootSessionID)
	assert.Equal(t, "/p", tv.ProjectPath)
	assert.Equal(t, 64000, tv.TokensMax)
	assert.Equal(t, 2, tv.ActiveClaims, "the root's claim and c2's")
	assert.Equal(t, 2, tv.QueuedClaims, "the seeded waiter and c1")
	assert.Equal(t, 10, tv.SearchCalls)
	assert.Equal(t, 5, tv.RepeatCalls)
	assert.InDelta(t, 0.5, tv.RepeatRate, 1e-9)
	assert.Equal(t, 550, tv.TokensDelivered)
	assert.Equal(t, 600, tv.TokensSaved, "a child that pulled more than its snapshot saves 0, not a negative")
	require.Len(t, tv.Handoffs, 1)
	h := tv.Handoffs[0]
	assert.Equal(t, newer.ref, h.Ref)
	assert.Equal(t, ModeFork, h.Mode)
	assert.Equal(t, SessionID("root-newer"), h.ParentSessionID)
	require.Len(t, h.Children, 2)
	byID := map[SessionID]ChildView{}
	for _, c := range h.Children {
		byID[c.SessionID] = c
	}
	assert.Equal(t, ChildView{
		SessionID: c1, Status: StatusDone, Depth: 1, OpenedAt: sqlTime(now.Add(-time.Hour)), LastActivityAt: sqlTime(now.Add(-time.Hour)),
		ResultRef: "ctx_r", Summary: "did it", SearchCalls: 4, RepeatCalls: 1, RepeatRate: 0.25,
		TokensAvailable: 900, TokensDelivered: 300, TokensSaved: 600, ActiveClaims: 0, QueuedClaims: 1,
	}, byID[c1])
	assert.Equal(t, 1, byID[c2].ActiveClaims)
	assert.Zero(t, byID[c2].TokensSaved)

	limited, err := TreeViews(1)
	require.NoError(t, err)
	require.Len(t, limited, 1)
	assert.Equal(t, newer.tree, limited[0].TreeID)
}
