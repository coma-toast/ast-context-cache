package memory

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// vectorOnlyQuery matches no subject, predicate, object, or rule, so FTS and
// LIKE both miss and Recall falls through to vectorSearch.
const vectorOnlyQuery = "qqqvectoronly"

// stubEmbedder returns fixed vectors keyed by text, so similarity to the query
// (and therefore rank) is chosen by the test.
type stubEmbedder map[string][]float32

func (s stubEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = s[t]
	}
	return out, nil
}

func (s stubEmbedder) EmbedSingle(text string) ([]float32, error) { return s[text], nil }

// unitVector has cosine similarity sim with unitVector(1).
func unitVector(sim float64) []float32 {
	v := make([]float32, search.VectorDims)
	v[0], v[1] = float32(sim), float32(math.Sqrt(1-sim*sim))
	return v
}

// TestVectorRecallValidityScopeAndRank covers BF-2: the vector fallback used to
// re-select rows by ref alone, returning superseded, forgotten, and
// out-of-scope entries in table order rather than similarity order.
func TestVectorRecallValidityScopeAndRank(t *testing.T) {
	dbtest.Init(t)
	search.Cache.Unload()
	t.Cleanup(search.Cache.Unload)
	emb := stubEmbedder{vectorOnlyQuery: unitVector(1)}
	store := func(in StoreInput) string {
		t.Helper()
		in.Kind, in.InvalidatePrevious = KindFact, true
		res, err := Store(in)
		require.NoError(t, err)
		return res.Ref
	}
	// embed keys ref's vector the way store_memory does: by the storing session.
	embed := func(ref, storingSession string, sim float64) {
		t.Helper()
		key := "vec-" + ref
		emb[key] = unitVector(sim)
		EmbedEntry(ref, storingSession, key, emb)
	}
	superseded := store(StoreInput{Scope: ScopeSession, SessionID: "S1", Subject: "db.engine", Object: "mysql"})
	current := store(StoreInput{Scope: ScopeSession, SessionID: "S1", Subject: "db.engine", Object: "postgres"})
	forgotten := store(StoreInput{Scope: ScopeSession, SessionID: "S1", Subject: "cache.ttl", Object: "5m"})
	res, err := Forget(ForgetInput{Refs: []string{forgotten}})
	require.NoError(t, err)
	require.Equal(t, []string{forgotten}, res.Invalidated)
	otherSession := store(StoreInput{Scope: ScopeSession, SessionID: "S2", Subject: "ui.theme", Object: "dark"})
	global := store(StoreInput{Scope: ScopeGlobal, Subject: "license", Object: "mit"})
	mid := store(StoreInput{Scope: ScopeSession, SessionID: "S1", Subject: "lint.tool", Object: "golangci"})
	high := store(StoreInput{Scope: ScopeSession, SessionID: "S1", Subject: "test.runner", Object: "make"})
	// Embedded in ascending similarity so table order disagrees with rank order.
	embed(current, "S1", 0.6)
	embed(global, "", 0.7)
	embed(mid, "S1", 0.8)
	embed(high, "S1", 0.9)
	embed(forgotten, "S1", 0.95)
	embed(superseded, "S1", 0.99)
	// Session-less key passes the vector layer for an unscoped recall; only the
	// SQL re-select can drop it.
	embed(otherSession, "", 0.97)
	tests := []struct {
		name string
		in   RecallInput
		want []string
	}{
		{name: "current in rank order", in: RecallInput{SessionID: "S1"}, want: []string{high, mid, global, current}},
		{name: "session scope skips global", in: RecallInput{SessionID: "S1", Scope: ScopeSession}, want: []string{high, mid, current}},
		{name: "history keeps invalidated", in: RecallInput{SessionID: "S1", IncludeHistory: true}, want: []string{superseded, forgotten, high, mid, global, current}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.in.Query = vectorOnlyQuery
			fts, err := searchFTS(tc.in)
			require.NoError(t, err)
			require.Empty(t, fts, "FTS must miss so the vector fallback runs")
			got, err := Recall(tc.in, emb)
			require.NoError(t, err)
			refs := make([]string, 0, len(got.Entries))
			for _, e := range got.Entries {
				refs = append(refs, e.Ref)
			}
			assert.Equal(t, tc.want, refs)
			assert.NotContains(t, refs, otherSession)
		})
	}
}

// TestVectorRecallCrossSessionProjectMemory covers BF-8: a project memory's vector is keyed
// by the session that stored it, so recall from another session used to drop it before the
// SQL re-select could apply the project scope.
func TestVectorRecallCrossSessionProjectMemory(t *testing.T) {
	dbtest.Init(t)
	search.Cache.Unload()
	t.Cleanup(search.Cache.Unload)
	emb := stubEmbedder{vectorOnlyQuery: unitVector(1)}
	store := func(in StoreInput, storingSession string, sim float64) string {
		t.Helper()
		in.Kind = KindFact
		res, err := Store(in)
		require.NoError(t, err)
		key := "vec-" + res.Ref
		emb[key] = unitVector(sim)
		EmbedEntry(res.Ref, storingSession, key, emb)
		return res.Ref
	}
	project := store(StoreInput{Scope: ScopeProject, SessionID: "S1", ProjectPath: "/proj", Subject: "build.cmd", Object: "make"}, "S1", 0.9)
	otherProject := store(StoreInput{Scope: ScopeProject, SessionID: "S1", ProjectPath: "/other", Subject: "build.cmd", Object: "just"}, "S1", 0.95)
	otherSession := store(StoreInput{Scope: ScopeSession, SessionID: "S1", Subject: "scratch", Object: "x"}, "S1", 0.8)
	tests := []struct {
		name string
		in   RecallInput
		want []string
	}{
		{name: "unscoped from another session", in: RecallInput{SessionID: "S2", ProjectPath: "/proj"}, want: []string{project}},
		{name: "project scope from another session", in: RecallInput{SessionID: "S2", ProjectPath: "/proj", Scope: ScopeProject}, want: []string{project}},
		{name: "session scope stays in session", in: RecallInput{SessionID: "S2", ProjectPath: "/proj", Scope: ScopeSession}},
		{name: "storing session sees its own entries", in: RecallInput{SessionID: "S1", ProjectPath: "/proj"}, want: []string{project, otherSession}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.in.Query = vectorOnlyQuery
			got, err := Recall(tc.in, emb)
			require.NoError(t, err)
			var refs []string
			for _, e := range got.Entries {
				refs = append(refs, e.Ref)
			}
			assert.Equal(t, tc.want, refs)
			assert.NotContains(t, refs, otherProject)
		})
	}
}

func TestValidityClause(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		in         RecallInput
		wantClause string
		wantArgs   []any
	}{
		{name: "now", in: RecallInput{}, wantClause: validNowClause},
		{name: "history", in: RecallInput{IncludeHistory: true}},
		{name: "as of", in: RecallInput{AsOf: "2026-01-01"}, wantClause: validAsOfClause, wantArgs: []any{"2026-01-01", "2026-01-01"}},
		{name: "history as of", in: RecallInput{AsOf: "2026-01-01", IncludeHistory: true}, wantClause: validFromAsOfClause, wantArgs: []any{"2026-01-01"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			clause, args := validityClause(tc.in, entryValidity)
			assert.Equal(t, tc.wantClause, clause)
			assert.Equal(t, tc.wantArgs, args)
		})
	}
}
