package hooks

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistryPendingFIFO(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	r := NewRegistry(t.TempDir())
	for _, ref := range []string{refA, refB} {
		require.NoError(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: ref}))
	}
	require.NoError(t, r.AddPending(ctx, "other-session", PendingHandoff{Ref: "hof_other"}))
	for _, want := range []string{refA, refB} {
		p, ok, err := r.TakePending(ctx, parentSID)
		require.NoError(t, err)
		require.True(t, ok)
		assert.Equal(t, want, p.Ref)
	}
	_, ok, err := r.TakePending(ctx, parentSID)
	require.NoError(t, err)
	assert.False(t, ok)
	p, ok, err := r.TakePending(ctx, "other-session")
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, "hof_other", p.Ref)
}

func TestRegistryDropsExpiredEntries(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	r := NewRegistry(t.TempDir())
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	r.now = func() time.Time { return now }
	require.NoError(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: refA, CreatedAt: now.Add(-2 * time.Hour)}))
	require.NoError(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: refB}))
	require.NoError(t, r.SetAgent(ctx, parentSID, "old", AgentHandoff{Ref: refA, SessionID: "s", CreatedAt: now.Add(-61 * time.Minute)}))
	require.NoError(t, r.SetAgent(ctx, parentSID, "new", AgentHandoff{Ref: refB, SessionID: "s2"}))
	p, ok, err := r.TakePending(ctx, parentSID)
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, refB, p.Ref, "the expired entry was dropped")
	_, ok, err = r.Agent(ctx, parentSID, "old")
	require.NoError(t, err)
	assert.False(t, ok)
	a, ok, err := r.TakeAgent(ctx, parentSID, "new")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "s2", a.SessionID)
	entries, err := os.ReadDir(r.dir)
	require.NoError(t, err)
	assert.Empty(t, entries, "an emptied session file and its lock are removed")
}

func TestRegistryBreaksStaleLock(t *testing.T) {
	t.Parallel()
	r := NewRegistry(t.TempDir())
	require.NoError(t, os.MkdirAll(r.dir, 0o700))
	lock := filepath.Join(r.dir, sessionKey(parentSID)+".json.lock")
	require.NoError(t, os.WriteFile(lock, nil, 0o600))
	old := time.Now().Add(-time.Minute)
	require.NoError(t, os.Chtimes(lock, old, old))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	require.NoError(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: refA}))
}

func TestRegistryLockTimesOut(t *testing.T) {
	t.Parallel()
	r := NewRegistry(t.TempDir())
	require.NoError(t, os.MkdirAll(r.dir, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(r.dir, sessionKey(parentSID)+".json.lock"), nil, 0o600))
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	assert.Error(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: refA}))
}

func TestRegistryCorruptFileReadsEmpty(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	r := NewRegistry(t.TempDir())
	require.NoError(t, os.MkdirAll(r.dir, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(r.dir, sessionKey(parentSID)+".json"), []byte("{oops"), 0o600))
	require.NoError(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: refA}))
	p, ok, err := r.TakePending(ctx, parentSID)
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, refA, p.Ref)
}

// TestRegistryConcurrentWriters runs parallel hooks against one session file, as parallel
// subagent spawns do; no write may be lost.
func TestRegistryConcurrentWriters(t *testing.T) {
	t.Parallel()
	const writers = 24
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	dir := t.TempDir()
	var wg sync.WaitGroup
	for i := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := NewRegistry(dir)
			ref := fmt.Sprintf("hof_%016x", i)
			assert.NoError(t, r.AddPending(ctx, parentSID, PendingHandoff{Ref: ref}))
			assert.NoError(t, r.SetAgent(ctx, parentSID, ref, AgentHandoff{Ref: ref, SessionID: ref + ".c1"}))
		}()
	}
	wg.Wait()
	r := NewRegistry(dir)
	seen := map[string]bool{}
	var mu sync.Mutex
	var takers sync.WaitGroup
	for range writers {
		takers.Add(1)
		go func() {
			defer takers.Done()
			p, ok, err := r.TakePending(ctx, parentSID)
			assert.NoError(t, err)
			assert.True(t, ok)
			_, ok, err = r.TakeAgent(ctx, parentSID, p.Ref)
			assert.NoError(t, err)
			assert.True(t, ok)
			mu.Lock()
			defer mu.Unlock()
			assert.False(t, seen[p.Ref], "each pending handoff is taken once")
			seen[p.Ref] = true
		}()
	}
	takers.Wait()
	assert.Len(t, seen, writers)
}
