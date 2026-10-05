package handoff

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTouchCoalescesAndRevives(t *testing.T) {
	s := newFanInService(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now.Add(-time.Hour))
	h := seedHandoff(t, handoffSeed{root: "parent", children: 2, at: now.Add(-time.Hour)})
	child := h.children[0]
	setClock(t, now)
	n, err := s.markAbandoned()
	require.NoError(t, err)
	require.Equal(t, 2, n)
	activity := func(sid SessionID) string {
		return queryString(t, `SELECT last_activity_at FROM handoff_children WHERE child_session_id = ?`, sid)
	}
	treeAccess := func() string {
		return queryString(t, `SELECT last_access_at FROM handoff_trees WHERE tree_id = ?`, h.tree)
	}

	wake := s.waiters.wait(h.tree)
	s.Touch(child)
	assert.Equal(t, StatusOpen, childStatus(t, child), "activity revives an abandoned child (FI-3)")
	assert.Equal(t, StatusAbandoned, childStatus(t, h.children[1]), "siblings untouched")
	assert.Equal(t, sqlTime(now), activity(child))
	assert.Equal(t, sqlTime(now), treeAccess())
	assert.Equal(t, sqlTime(now), queryString(t, `SELECT last_access_at FROM handoffs WHERE ref = ?`, h.ref))
	assert.True(t, woken(wake), "collect waiters see the status change")

	setClock(t, now.Add(5*time.Second))
	s.Touch(child)
	assert.Equal(t, sqlTime(now), activity(child), "a second touch inside 10s is coalesced")
	s.Touch(h.children[1])
	assert.Equal(t, StatusOpen, childStatus(t, h.children[1]), "coalescing is per session")

	setClock(t, now.Add(11*time.Second))
	wake = s.waiters.wait(h.tree)
	s.Touch(child)
	assert.Equal(t, sqlTime(now.Add(11*time.Second)), activity(child), "written again after the window")
	assert.Equal(t, sqlTime(now.Add(11*time.Second)), treeAccess())
	assert.False(t, woken(wake), "no status change, no wake-up")
}

func TestTouchIgnoresNonChildren(t *testing.T) {
	s := newFanInService(t)
	at := time.Date(2026, 10, 5, 11, 0, 0, 0, time.UTC)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 1, at: at})
	setClock(t, at.Add(time.Hour))
	s.Touch("")
	s.Touch("stranger")
	s.Touch(h.root)
	assert.Equal(t, sqlTime(at), queryString(t, `SELECT last_access_at FROM handoff_trees WHERE tree_id = ?`, h.tree), "roots and strangers write nothing")

	// A child whose tree was flushed after the index learned it is a no-op, not an error.
	require.True(t, s.IsTreeSession(h.children[0]))
	exec(t, `DELETE FROM handoff_children WHERE child_session_id = ?`, h.children[0])
	s.Touch(h.children[0])
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_children WHERE child_session_id = ?`, h.children[0]))
}

func TestTouchCoalescerBounded(t *testing.T) {
	t.Parallel()
	c := newTouchCoalescer()
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	for i := range maxTouchEntries {
		require.True(t, c.due(time.Duration(i).String(), now))
	}
	assert.True(t, c.due("late", now.Add(touchWindow)))
	assert.Len(t, c.last, 1, "stale entries are dropped once the map is full")
	assert.False(t, c.due("late", now.Add(touchWindow+time.Second)))
	c.forget("late")
	assert.True(t, c.due("late", now.Add(touchWindow+time.Second)), "a forgotten key writes again")
	assert.True(t, c.due("late", now), "a clock that went back writes")
}
