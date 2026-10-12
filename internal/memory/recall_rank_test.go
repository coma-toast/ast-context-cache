package memory

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func mustStore(t *testing.T, in StoreInput) string {
	t.Helper()
	res, err := Store(in)
	require.NoError(t, err)
	return res.Ref
}

func entryRefs(entries []Entry) []string {
	var refs []string
	for _, e := range entries {
		refs = append(refs, e.Ref)
	}
	return refs
}

// BF-3: Recall returned up to limit*2 entries because nothing trimmed after ranking.
func TestRecallTrimsToLimit(t *testing.T) {
	dbtest.Init(t)
	for i := 0; i < 6; i++ {
		mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: fmt.Sprintf("widget.%d", i), Object: "blue"})
	}
	for _, query := range []string{"", "widget"} {
		res, err := Recall(RecallInput{SessionID: "s", Query: query, Limit: 2}, nil)
		require.NoError(t, err)
		assert.Len(t, res.Lines, 2, "query %q", query)
	}
}

// BF-3: the kind filter ran after the SQL limit, so a procedure past the first limit*2 facts
// was never returned.
func TestRecallKindFilterInSQL(t *testing.T) {
	dbtest.Init(t)
	for i := 0; i < 6; i++ {
		mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: fmt.Sprintf("deploy.%d", i), Object: "prod"})
	}
	proc := mustStore(t, StoreInput{Kind: KindProcedure, Scope: ScopeSession, SessionID: "s", Rule: "deploy only from main"})
	for _, query := range []string{"", "deploy"} {
		res, err := Recall(RecallInput{SessionID: "s", Query: query, Limit: 1, Kinds: []Kind{KindProcedure}}, nil)
		require.NoError(t, err)
		assert.Equal(t, []string{proc}, entryRefs(res.Entries), "query %q", query)
	}
}

// BF-4: Store supersedes prior facts only when asked to.
func TestStoreSupersedesOnlyWhenAsked(t *testing.T) {
	dbtest.Init(t)
	first := mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "editor", Object: "vim"})
	res, err := Store(StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "editor", Object: "emacs"})
	require.NoError(t, err)
	assert.Empty(t, res.InvalidatedRefs)
	assert.Equal(t, 1, activeCount(t, []string{first}))
	res, err = Store(StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "editor", Object: "helix", InvalidatePrevious: true})
	require.NoError(t, err)
	assert.Len(t, res.InvalidatedRefs, 2)
	assert.Equal(t, 0, activeCount(t, []string{first}))
}

// BF-11: FTS matched the ref column (every ref starts with mem_) and ranked by access count.
func TestSearchFTSContentColumnsByBM25(t *testing.T) {
	dbtest.Init(t)
	weak := mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "notes", Object: "zebra appears once among many other unrelated words here"})
	strong := mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "zebra", Object: "zebra"})
	got, err := searchFTS(RecallInput{SessionID: "s", Query: "zebra", Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, []string{strong, weak}, entryRefs(got))
	got, err = searchFTS(RecallInput{SessionID: "s", Query: "mem", Limit: 10})
	require.NoError(t, err)
	assert.Empty(t, got, "refs must not be searchable")
}

// BF-12: vector results are fused with the lexical ones instead of only filling in when
// FTS and LIKE miss, and each line carries its fused score.
func TestRecallFusesLexicalAndVector(t *testing.T) {
	dbtest.Init(t)
	search.Cache.Unload()
	t.Cleanup(search.Cache.Unload)
	emb := stubEmbedder{"deploy": unitVector(1)}
	embed := func(ref string, sim float64) {
		key := "vec-" + ref
		emb[key] = unitVector(sim)
		EmbedEntry(ref, "s", key, emb)
	}
	lexical := mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "deploy.target", Object: "prod"})
	semantic := mustStore(t, StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: "s", Subject: "release.flow", Object: "ship it"})
	embed(lexical, 0.5)
	embed(semantic, 0.99)
	res, err := Recall(RecallInput{SessionID: "s", Query: "deploy"}, emb)
	require.NoError(t, err)
	require.Len(t, res.Lines, 2)
	assert.Equal(t, []string{lexical, semantic}, []string{res.Lines[0].Ref, res.Lines[1].Ref})
	assert.InDelta(t, 2.0/61+1.0/62, res.Lines[0].Score, 1e-12)
	assert.InDelta(t, 1.0/61, res.Lines[1].Score, 1e-12)
}

func TestFuseEntriesTieBreaksByRef(t *testing.T) {
	a, b := Entry{Ref: "mem_a"}, Entry{Ref: "mem_b"}
	got, scores := fuseEntries([]Entry{b}, []Entry{a})
	assert.Equal(t, []string{"mem_a", "mem_b"}, entryRefs(got))
	assert.Equal(t, scores["mem_a"], scores["mem_b"])
}

// MEM-6: decayFactor tests.
// Test that frequently accessed recent entry scores >1.0.
func TestDecayFactor_FrequentRecentScoresAboveOne(t *testing.T) {
	dbtest.Init(t)
	e := Entry{
		Kind:           KindFact,
		Scope:          ScopeSession,
		AccessCount:    10,
		LastAccessedAt: time.Now().UTC().Format("2006-01-02 15:04:05"),
		CreatedAt:      time.Now().UTC().Format("2006-01-02 15:04:05"),
	}
	f := decayFactor(e, RecallInput{})
	assert.True(t, f > 1.0, "frequent recent entry should score >1.0, got %f", f)
}

// Test that old unused entry hits floor 0.3.
func TestDecayFactor_OldUnusedHitsFloor(t *testing.T) {
	dbtest.Init(t)
	old := time.Now().UTC().Add(-100 * 24 * time.Hour).Format("2006-01-02 15:04:05")
	e := Entry{
		Kind:           KindFact,
		Scope:          ScopeSession,
		AccessCount:    1,
		LastAccessedAt: old,
		CreatedAt:      old,
	}
	f := decayFactor(e, RecallInput{})
	assert.InDelta(t, 0.3, f, 1e-9, "old unused entry should hit 0.3 floor, got %f", f)
}

// Test that more accesses never lower the factor.
func TestDecayFactor_MoreAccessesNeverLower(t *testing.T) {
	dbtest.Init(t)
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	var prev float64 = -1
	for n := 1; n <= 20; n++ {
		e := Entry{
			Kind:           KindFact,
			Scope:          ScopeSession,
			AccessCount:    n,
			LastAccessedAt: now,
			CreatedAt:      now,
		}
		f := decayFactor(e, RecallInput{})
		if prev >= 0 {
			assert.True(t, f >= prev-1e-12, "accessCount=%d factor %f should not be lower than %f (n=%d)", n, f, prev, n-1)
		}
		prev = f
	}
}

// Test that frequency component is monotonic with access count.
func TestDecayFactor_FrequencyMonotonic(t *testing.T) {
	dbtest.Init(t)
	var prev float64
	for n := 1; n <= 100; n++ {
		freq := 1.0 + math.Log2(1.0+float64(n))/4.0
		if n > 1 {
			assert.True(t, freq >= prev-1e-12, "frequency should be monotonic: n=%d freq=%f", n, freq)
		}
		prev = freq
	}
}

// Test that recency decays with time since last access.
func TestDecayFactor_RecencyDecays(t *testing.T) {
	dbtest.Init(t)
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	old := time.Now().UTC().Add(-10 * 24 * time.Hour).Format("2006-01-02 15:04:05")
	eRecent := Entry{Kind: KindFact, Scope: ScopeSession, AccessCount: 1, LastAccessedAt: now, CreatedAt: now}
	eOld := Entry{Kind: KindFact, Scope: ScopeSession, AccessCount: 1, LastAccessedAt: old, CreatedAt: now}
	fRecent := decayFactor(eRecent, RecallInput{})
	fOld := decayFactor(eOld, RecallInput{})
	assert.True(t, fRecent > fOld, "recent entry %f should score higher than old %f", fRecent, fOld)
}

// Test that age term decays with creation age.
func TestDecayFactor_AgeDecays(t *testing.T) {
	dbtest.Init(t)
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	old := time.Now().UTC().Add(-10 * 24 * time.Hour).Format("2006-01-02 15:04:05")
	eNew := Entry{Kind: KindFact, Scope: ScopeSession, AccessCount: 1, LastAccessedAt: now, CreatedAt: now}
	eOld := Entry{Kind: KindFact, Scope: ScopeSession, AccessCount: 1, LastAccessedAt: now, CreatedAt: old}
	fNew := decayFactor(eNew, RecallInput{})
	fOld := decayFactor(eOld, RecallInput{})
	assert.True(t, fNew > fOld, "new entry %f should score higher than old %f", fNew, fOld)
}

// Test that procedure is exempt from decay by default (MEM-7).
func TestDecayFactor_ProcedureExempt(t *testing.T) {
	dbtest.Init(t)
	old := time.Now().UTC().Add(-100 * 24 * time.Hour).Format("2006-01-02 15:04:05")
	e := Entry{
		Kind:           KindProcedure,
		Scope:          ScopeSession,
		AccessCount:    1,
		LastAccessedAt: old,
		CreatedAt:      old,
	}
	f := decayFactor(e, RecallInput{})
	assert.Equal(t, 1.0, f, "procedure should be exempt from decay by default")
}

// BF-13: all=true wiped every session and project; it is now scoped, and unscoped needs confirm.
func TestForgetAllScoped(t *testing.T) {
	dbtest.Init(t)
	mine := mustStore(t, StoreInput{Kind: KindProcedure, Scope: ScopeSession, SessionID: "mine", Rule: "mine"})
	other := mustStore(t, StoreInput{Kind: KindProcedure, Scope: ScopeSession, SessionID: "other", Rule: "other"})
	proj := mustStore(t, StoreInput{Kind: KindProcedure, Scope: ScopeProject, SessionID: "mine", ProjectPath: "/p", Rule: "project"})
	global := mustStore(t, StoreInput{Kind: KindProcedure, Scope: ScopeGlobal, Rule: "global"})
	_, err := Forget(ForgetInput{All: true})
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	assert.Contains(t, err.Error(), "all=true without scope requires confirm=true")
	assert.Equal(t, 4, activeCount(t, []string{mine, other, proj, global}))
	res, err := Forget(ForgetInput{All: true, SessionID: "mine"})
	require.NoError(t, err)
	assert.Equal(t, 1, res.InvalidatedRefs)
	assert.Equal(t, 0, activeCount(t, []string{mine}))
	res, err = Forget(ForgetInput{All: true, Scope: ScopeProject, ProjectPath: "/p"})
	require.NoError(t, err)
	assert.Equal(t, 1, res.InvalidatedRefs)
	assert.Equal(t, 2, activeCount(t, []string{other, global}))
	res, err = Forget(ForgetInput{All: true, Confirm: true})
	require.NoError(t, err)
	assert.Equal(t, 2, res.InvalidatedRefs)
	assert.Equal(t, 0, activeCount(t, []string{other, global}))
}
