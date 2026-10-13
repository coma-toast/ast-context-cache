package memory

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHistoryAfterSupersede(t *testing.T) {
	testMemoryDB(t)

	// Store A
	resA, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "history.supersede",
		Predicate:          "is",
		Object:             "v1",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Store B (supersedes A)
	resB, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "history.supersede",
		Predicate:          "is",
		Object:             "v2",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Store C (supersedes B)
	resC, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "history.supersede",
		Predicate:          "is",
		Object:             "v3",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// History(ref) returns all versions (including superseded)
	result, err := History(HistoryInput{Ref: resA.Ref, IncludeWorld: false})
	require.NoError(t, err)
	assert.Len(t, result.Entries, 3, "should see all 3 versions in the chain")

	// Verify entries have correct supersession chain
	refs := make(map[string]bool)
	for _, e := range result.Entries {
		refs[e.Ref] = true
	}
	assert.True(t, refs[resA.Ref])
	assert.True(t, refs[resB.Ref])
	assert.True(t, refs[resC.Ref])

	// Verify chain ordering: A was superseded by B, B by C
	var foundA, foundB bool
	for _, e := range result.Entries {
		if e.Ref == resA.Ref {
			foundA = true
			assert.Equal(t, resB.Ref, e.SupersededBy, "A's superseded_by should be B")
		}
		if e.Ref == resB.Ref {
			foundB = true
			assert.Equal(t, resC.Ref, e.SupersededBy, "B's superseded_by should be C")
		}
	}
	assert.True(t, foundA, "A should be in history")
	assert.True(t, foundB, "B should be in history")
}

func TestHistoryByQuery(t *testing.T) {
	testMemoryDB(t)

	// Store entries with different subjects
	_, err := Store(StoreInput{
		Kind:      KindFact,
		Scope:     ScopeGlobal,
		Subject:   "history.q1",
		Predicate: "is",
		Object:    "active",
	})
	require.NoError(t, err)

	_, err = Store(StoreInput{
		Kind:      KindFact,
		Scope:     ScopeGlobal,
		Subject:   "history.q1",
		Predicate: "is",
		Object:    "superseded",
	})
	require.NoError(t, err)

	_, err = Store(StoreInput{
		Kind:      KindFact,
		Scope:     ScopeGlobal,
		Subject:   "history.q2",
		Predicate: "is",
		Object:    "match",
	})
	require.NoError(t, err)

	// Query for "history.q1" should return only q1 entries
	result, err := History(HistoryInput{Query: "history.q1", IncludeWorld: false})
	require.NoError(t, err)
	assert.Len(t, result.Entries, 2, "should return both q1 entries")

	// Query for "history.q2" should return only q2 entries
	result, err = History(HistoryInput{Query: "history.q2", IncludeWorld: false})
	require.NoError(t, err)
	assert.Len(t, result.Entries, 1, "should return one q2 entry")

	// Query for "match" should return q2 entry (object contains "match")
	result, err = History(HistoryInput{Query: "match", IncludeWorld: false})
	require.NoError(t, err)
	assert.NotEmpty(t, result.Entries, "should find entry with object=match")
}

func TestHistoryNotFound(t *testing.T) {
	testMemoryDB(t)

	_, err := History(HistoryInput{Ref: "nonexistent-ref", IncludeWorld: false})
	assert.Error(t, err, "should error for nonexistent ref")
}
