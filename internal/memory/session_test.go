package memory

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func storeEntry(t *testing.T, in StoreInput) string {
	t.Helper()
	res, err := Store(in)
	require.NoError(t, err)
	return res.Ref
}

func TestActiveForSession(t *testing.T) {
	testMemoryDB(t)
	first := storeEntry(t, StoreInput{Kind: KindFact, SessionID: "act-s", Subject: "api", Predicate: "uses", Object: "grpc"})
	rule := storeEntry(t, StoreInput{Kind: KindProcedure, SessionID: "act-s", Rule: "run make lint"})
	storeEntry(t, StoreInput{Kind: KindFact, SessionID: "other-s", Subject: "api", Predicate: "uses", Object: "rest"})
	storeEntry(t, StoreInput{Kind: KindFact, Scope: ScopeGlobal, Subject: "user", Predicate: "likes", Object: "go"})
	forgotten := storeEntry(t, StoreInput{Kind: KindProcedure, SessionID: "act-s", Rule: "old rule"})
	_, err := Forget(ForgetInput{Refs: []string{forgotten}, SessionID: "act-s"})
	require.NoError(t, err)
	entries, err := ActiveForSession("act-s")
	require.NoError(t, err)
	var refs []string
	for _, e := range entries {
		refs = append(refs, e.Ref)
		assert.Equal(t, ScopeSession, e.Scope)
	}
	assert.ElementsMatch(t, []string{first, rule}, refs)
	var access int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COALESCE(SUM(access_count),0) FROM structured_memory`).Scan(&access))
	assert.Zero(t, access, "no access accounting")
	_, err = ActiveForSession("")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

func TestDeleteSession(t *testing.T) {
	testMemoryDB(t)
	a := storeEntry(t, StoreInput{Kind: KindFact, SessionID: "del-s", Subject: "db", Predicate: "is", Object: "sqlite"})
	b := storeEntry(t, StoreInput{Kind: KindProcedure, SessionID: "del-s", Rule: "keep it compact"})
	_, err := Forget(ForgetInput{Refs: []string{b}, SessionID: "del-s"})
	require.NoError(t, err)
	keep := storeEntry(t, StoreInput{Kind: KindFact, SessionID: "keep-s", Subject: "db", Predicate: "is", Object: "postgres"})
	key := memoryVectorKey(a)
	vec := make([]float32, search.VectorDims)
	vec[0] = 1
	require.NoError(t, search.Cache.Upsert([]search.VectorEntry{{
		ContentHash: search.ContentHash(key), DocType: "memory", SourceFile: key, Name: key, ProjectPath: "del-s", Vector: vec,
	}}))
	n, err := DeleteSession("del-s")
	require.NoError(t, err)
	assert.Equal(t, 2, n, "current and forgotten rows both go")
	assertGone(t, a)
	assertGone(t, b)
	assertPresent(t, keep)
	assert.Zero(t, memoryVectorRows(t, key))
	n, err = DeleteSession("del-s")
	require.NoError(t, err)
	assert.Zero(t, n)
}
