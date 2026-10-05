package trail

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

// DB-backed tests do not use t.Parallel: the db pools and the in-memory rings are package globals.

func setup(t *testing.T) {
	t.Helper()
	dbtest.Init(t)
	// Write buffers outlive Init/Close: flush this test's rows into its own DB before it closes.
	t.Cleanup(db.FlushWriteBuffers)
	resetMemory()
	t.Cleanup(resetMemory)
}

func entry(sid, query string, hits int) Entry {
	return Entry{SessionID: sid, Tool: "get_context_capsule", Query: query, Mode: "auto", ProjectPath: "/p", HitCount: hits}
}

func TestNormalizeQuery(t *testing.T) {
	t.Parallel()
	tests := []struct{ in, want string }{
		{"Auth Handler", "auth handler"},
		{"  auth\t\thandler \n", "auth handler"},
		{"", ""},
		{"   ", ""},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, NormalizeQuery(tt.in))
		})
	}
}

func TestMatchKey(t *testing.T) {
	t.Parallel()
	e := Entry{Tool: "search_semantic", QueryNorm: "auth handler", FiltersKey: "p:x|k:|l:go", DocType: "code"}
	assert.Equal(t, "search_semantic|auth handler|p:x|k:|l:go|code", e.MatchKey())
	assert.Equal(t, "a.go#Run@12", HitRef("a.go", "Run", 12))
}

func TestRecordVisibleImmediately(t *testing.T) {
	setup(t)
	e := entry("s1", "  Auth   Handler ", 0)
	e.TopHits = []string{"a#1@1", "b#2@2", "c#3@3", "d#4@4", "e#5@5", "f#6@6"}
	e.CandidateHits = []string{"a#1@1"}
	Record(e)
	got := ForSession("s1", 10)
	require.Len(t, got, 1)
	assert.Equal(t, "auth handler", got[0].QueryNorm)
	assert.True(t, got[0].ZeroHit)
	assert.Len(t, got[0].TopHits, MaxTopHits)
	assert.False(t, got[0].At.IsZero())
	assert.Nil(t, got[0].CandidateHits, "candidate hits are never stored")
	assert.Empty(t, ForSession("other", 10))
	assert.Empty(t, ForSession("", 10))
	Record(Entry{Query: "no session"})
	assert.Empty(t, ForSession("", 10))
}

func TestPersistedRowsSurviveRingReset(t *testing.T) {
	setup(t)
	e := entry("s1", "load model", 3)
	e.TopHits = []string{"a.py#load@1"}
	e.FiltersKey, e.DocType = "p:x|k:|l:python", "code"
	Record(e)
	want := ForSession("s1", 10)
	db.FlushWriteBuffers()
	resetMemory()
	got := ForSession("s1", 10)
	require.Len(t, got, 1)
	assert.Equal(t, want, got)
	// A row both in memory and in the DB is returned once.
	Record(entry("s1", "unload", 1))
	db.FlushWriteBuffers()
	assert.Len(t, ForSession("s1", 10), 2)
}

func TestForSessionNewestFirstAndLimit(t *testing.T) {
	setup(t)
	base := time.Now().Add(-time.Hour)
	for i := 0; i < 5; i++ {
		e := entry("s1", fmt.Sprintf("q%d", i), i)
		e.At = base.Add(time.Duration(i) * time.Minute)
		Record(e)
		if i == 1 {
			// Older entries persisted and dropped from memory still merge in order.
			db.FlushWriteBuffers()
			resetMemory()
		}
	}
	got := ForSession("s1", 0)
	require.Len(t, got, 5)
	for i, e := range got {
		assert.Equal(t, fmt.Sprintf("q%d", 4-i), e.Query)
	}
	got = ForSession("s1", 2)
	require.Len(t, got, 2)
	assert.Equal(t, "q4", got[0].Query)
	assert.Equal(t, "q3", got[1].Query)
}

func TestLookup(t *testing.T) {
	setup(t)
	old := entry("s1", "Auth", 1)
	old.At = time.Now().Add(-time.Minute)
	Record(old)
	Record(entry("s1", "other", 2))
	Record(entry("s1", "auth ", 4))
	key := entry("", "", 0)
	key.QueryNorm = "auth"
	got, ok := Lookup("s1", key.MatchKey())
	require.True(t, ok)
	assert.Equal(t, 4, got.HitCount, "newest match wins")
	_, ok = Lookup("s1", "nope")
	assert.False(t, ok)
	_, ok = Lookup("s2", key.MatchKey())
	assert.False(t, ok)
	db.FlushWriteBuffers()
	resetMemory()
	got, ok = Lookup("s1", key.MatchKey())
	require.True(t, ok, "falls back to persisted rows")
	assert.Equal(t, 4, got.HitCount)
}

func TestPruneOlderThan(t *testing.T) {
	setup(t)
	old := entry("s1", "old", 1)
	old.At = time.Now().Add(-48 * time.Hour)
	Record(old)
	gone := entry("s2", "gone", 1)
	gone.At = old.At
	Record(gone)
	Record(entry("s1", "new", 1))
	db.FlushWriteBuffers()
	n, err := PruneOlderThan(24 * time.Hour)
	require.NoError(t, err)
	assert.Equal(t, int64(2), n)
	got := ForSession("s1", 10)
	require.Len(t, got, 1)
	assert.Equal(t, "new", got[0].Query)
	assert.Empty(t, ForSession("s2", 10), "ring entries are pruned too")
	_, ok := rings.Load("s2")
	assert.False(t, ok, "an emptied ring is dropped")
}

func TestSubscribe(t *testing.T) {
	setup(t)
	t.Cleanup(func() {
		subMu.Lock()
		subscribers = nil
		subMu.Unlock()
	})
	var got []Entry
	Subscribe(func(e Entry) { got = append(got, e) })
	e := entry("s1", "Q", 0)
	e.CandidateHits = []string{"a#b@1"}
	Record(e)
	Record(Entry{Query: "no session"})
	require.Len(t, got, 1)
	assert.Equal(t, "q", got[0].QueryNorm)
	assert.True(t, got[0].ZeroHit)
	assert.Equal(t, []string{"a#b@1"}, got[0].CandidateHits, "subscribers see candidate hits")
}

func TestRingCap(t *testing.T) {
	setup(t)
	base := time.Now().Add(-time.Hour)
	for i := 0; i < ringCap+10; i++ {
		e := entry("s1", fmt.Sprintf("q%d", i), 1)
		e.At = base.Add(time.Duration(i) * time.Millisecond)
		Record(e)
	}
	r, ok := rings.Load("s1")
	require.True(t, ok)
	mem := r.(*ring).newestFirst()
	require.Len(t, mem, ringCap)
	assert.Equal(t, fmt.Sprintf("q%d", ringCap+9), mem[0].Query)
	assert.Equal(t, "q10", mem[ringCap-1].Query)
	db.FlushWriteBuffers()
	assert.Len(t, ForSession("s1", 1000), ringCap+10, "evicted entries are still read from the DB")
}
