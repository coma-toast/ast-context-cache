package embedqueue

import (
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// resetQueueCaps drops the cap settings rows and restores the default soft limits.
func resetQueueCaps(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		for _, c := range queueCaps {
			_, _ = db.DB.Exec(`DELETE FROM settings WHERE key = ?`, c.Key)
		}
		LoadQueueCaps()
	})
}

func testJob(i int) job {
	return job{file: "/tmp/cap" + strconv.Itoa(i) + ".go", projectPath: "/proj-caps"}
}

func TestTrySendEnforcesSoftLimit(t *testing.T) {
	tests := []struct {
		name       string
		limit, buf int
		wantQueued int
	}{
		{name: "soft below buffer", limit: 2, buf: 8, wantQueued: 2},
		{name: "soft equals buffer", limit: 4, buf: 4, wantQueued: 4},
		{name: "buffer below soft", limit: 8, buf: 3, wantQueued: 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ch, g := make(chan job, tc.buf), newGate(tc.limit)
			n := 0
			for i := 0; i < 10; i++ {
				if trySend(ch, g, testJob(i)) {
					n++
				}
			}
			assert.Equal(t, tc.wantQueued, n)
			assert.Len(t, ch, tc.wantQueued)
		})
	}
	assert.False(t, trySend(nil, newGate(1), testJob(0)), "nil channel never accepts")
}

// A full pending retry channel (at its soft limit) leaves the job in pending, exactly as a
// full buffered channel did.
func TestPendingRetryAtSoftLimitStaysPending(t *testing.T) {
	resetQueueCaps(t)
	_, err := SetQueueCap(PendingCapSetting, "1")
	require.NoError(t, err)
	ch := make(chan job, 8)
	a, b := testJob(1), testJob(2)
	pendingMu.Lock()
	if pending == nil {
		pending = map[string]job{}
	}
	pending[jobKey(a)], pending[jobKey(b)] = a, b
	pendingMu.Unlock()
	t.Cleanup(func() {
		pendingMu.Lock()
		delete(pending, jobKey(a))
		delete(pending, jobKey(b))
		delete(pendingChQueued, jobKey(a))
		delete(pendingChQueued, jobKey(b))
		pendingMu.Unlock()
	})
	assert.True(t, enqueuePendingRetryOn(ch, a))
	assert.False(t, enqueuePendingRetryOn(ch, b), "second retry must not pass the soft limit")
	assert.Len(t, ch, 1)
	pendingMu.Lock()
	_, stillPending := pending[jobKey(b)]
	_, marked := pendingChQueued[jobKey(b)]
	pendingMu.Unlock()
	assert.True(t, stillPending)
	assert.False(t, marked, "rejected retry must be unmarked so a later flush retries it")
}

// sendWait blocks at the soft limit (as a send on a full channel did) and resumes on dequeue.
func TestSendWaitBlocksAtSoftLimitUntilDequeue(t *testing.T) {
	ch, g := make(chan job, 16), newGate(2)
	require.True(t, trySend(ch, g, testJob(1)))
	require.True(t, trySend(ch, g, testJob(2)))
	done := make(chan struct{})
	go func() {
		sendWait(testJob(3), lane{ch, g})
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("sendWait returned while the queue was at its soft limit")
	case <-time.After(50 * time.Millisecond):
	}
	<-ch
	signalSpace()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("sendWait did not resume after a dequeue")
	}
	assert.Len(t, ch, 2)
}

// pushHigh-style lanes overflow to background when priority is at its soft limit.
func TestSendWaitOverflowsToNextLane(t *testing.T) {
	high, low := make(chan job, 16), make(chan job, 16)
	hg, lg := newGate(1), newGate(4)
	sendWait(testJob(1), lane{high, hg}, lane{low, lg})
	sendWait(testJob(2), lane{high, hg}, lane{low, lg})
	assert.Len(t, high, 1)
	assert.Len(t, low, 1)
}

// Raising a limit from the settings path wakes a blocked sender without any dequeue or restart.
func TestSetQueueCapAppliesLive(t *testing.T) {
	resetQueueCaps(t)
	_, err := SetQueueCap(LowCapSetting, "2")
	require.NoError(t, err)
	assert.Equal(t, 2, Snapshot().LowCap)
	ch := make(chan job, 16)
	require.True(t, trySend(ch, lowGate, testJob(1)))
	require.True(t, trySend(ch, lowGate, testJob(2)))
	done := make(chan struct{})
	go func() {
		sendWait(testJob(3), lane{ch, lowGate})
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("sendWait returned at the soft limit")
	case <-time.After(50 * time.Millisecond):
	}
	_, err = SetQueueCap(LowCapSetting, "3")
	require.NoError(t, err)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("raising the cap did not unblock the sender")
	}
	assert.Equal(t, 3, Snapshot().LowCap)
	assert.Equal(t, "3", db.GetSetting(LowCapSetting, ""), "cap is persisted")
}

func TestQueueCapDefaultsAndPersistedLoad(t *testing.T) {
	resetQueueCaps(t)
	LoadQueueCaps()
	s := Snapshot()
	assert.Equal(t, []int{DefaultPendingCap, DefaultHighCap, DefaultLowCap}, []int{s.PendingQueueCap, s.HighCap, s.LowCap})
	require.NoError(t, db.SetSetting(HighCapSetting, "300"))
	require.NoError(t, db.SetSetting(LowCapSetting, strconv.Itoa(LowCapHardMax+5)))
	LoadQueueCaps()
	s = Snapshot()
	assert.Equal(t, 300, s.HighCap)
	assert.Equal(t, LowCapHardMax, s.LowCap, "a stored value above the hard max is clamped")
}

func TestQueueCapEnvLock(t *testing.T) {
	resetQueueCaps(t)
	t.Setenv("EMBED_QUEUE_LOW_CAP", "4")
	LoadQueueCaps()
	assert.Equal(t, 4, Snapshot().LowCap)
	_, err := SetQueueCap(LowCapSetting, "10")
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeConflict))
	assert.Contains(t, err.Error(), "EMBED_QUEUE_LOW_CAP")
	assert.Equal(t, 4, Snapshot().LowCap, "env-locked cap is unchanged")
	for _, st := range QueueCapStates() {
		assert.Equal(t, st.Key == LowCapSetting, st.EnvLocked, st.Key)
	}
}

func TestValidateQueueCapBounds(t *testing.T) {
	tests := []struct {
		key, raw string
		want     int
		code     errs.Code
	}{
		{PendingCapSetting, "1", 1, ""},
		{PendingCapSetting, strconv.Itoa(PendingCapHardMax), PendingCapHardMax, ""},
		{PendingCapSetting, strconv.Itoa(PendingCapHardMax + 1), 0, errs.CodeInvalidInput},
		{HighCapSetting, " 256 ", 256, ""},
		{HighCapSetting, "0", 0, errs.CodeInvalidInput},
		{HighCapSetting, "-3", 0, errs.CodeInvalidInput},
		{LowCapSetting, strconv.Itoa(LowCapHardMax), LowCapHardMax, ""},
		{LowCapSetting, "abc", 0, errs.CodeInvalidInput},
		{LowCapSetting, "", 0, errs.CodeInvalidInput},
		{"embed_queue_bogus_cap", "5", 0, errs.CodeNotFound},
	}
	for _, tc := range tests {
		t.Run(tc.key+"="+tc.raw, func(t *testing.T) {
			n, err := ValidateQueueCap(tc.key, tc.raw)
			if tc.code != "" {
				require.Error(t, err)
				assert.True(t, errs.HasCode(err, tc.code), err.Error())
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, n)
		})
	}
}
