package memory

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestRestoreReactivatesAndInvalidatesSuccessor(t *testing.T) {
	testMemoryDB(t)

	// Store A
	resA, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "restore.test",
		Predicate:          "is",
		Object:             "old",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Store B (supersedes A)
	resB, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "restore.test",
		Predicate:          "is",
		Object:             "new",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Verify: A should be invalidated, B should be active
	eA, err := entryByRef(resA.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, eA.ValidUntil, "A should be superseded")
	assert.Equal(t, resB.Ref, eA.SupersededBy)

	eB, err := entryByRef(resB.Ref)
	require.NoError(t, err)
	assert.Empty(t, eB.ValidUntil, "B should be active")

	// Restore A
	result, err := Restore(RestoreInput{Refs: []string{resA.Ref}})
	require.NoError(t, err)
	assert.Contains(t, result.RestoredRefs, resA.Ref, "A should be restored")
	assert.Empty(t, result.Conflicts)

	// Verify: A should be active again
	eA, err = entryByRef(resA.Ref)
	require.NoError(t, err)
	assert.Empty(t, eA.ValidUntil, "A should be active after restore")
	assert.Empty(t, eA.InvalidatedReason)
	assert.Empty(t, eA.SupersededBy)

	// Verify: B should now be invalidated with ReasonRestored
	eB, err = entryByRef(resB.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, eB.ValidUntil, "B should be invalidated after A is restored")
	assert.Equal(t, ReasonRestored, eB.InvalidatedReason, "B's invalidation reason should be ReasonRestored")
	assert.Equal(t, resA.Ref, eB.SupersededBy)

	// Verify: A is searchable (FTS reindexed)
	var count int
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory_fts WHERE ref = ?`, resA.Ref).Scan(&count)
	assert.Equal(t, 1, count, "restored A should be in FTS")
}

func TestRestoreConflictWhenSupersederWasSupersededAgain(t *testing.T) {
	testMemoryDB(t)

	// Store A
	resA, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "restore.conflict",
		Predicate:          "is",
		Object:             "first",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Store B (supersedes A)
	resB, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "restore.conflict",
		Predicate:          "is",
		Object:             "second",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Store C (supersedes B, making B invalid)
	resC, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "restore.conflict",
		Predicate:          "is",
		Object:             "third",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Try to restore A: B was superseded by C, so restoring A must report a
	// conflict and change nothing (MEM-4).
	result, err := Restore(RestoreInput{Refs: []string{resA.Ref}})
	require.NoError(t, err)
	assert.Empty(t, result.RestoredRefs, "A should not be restored")
	assert.NotEmpty(t, result.Conflicts, "should report a conflict when the superseding fact was superseded again")

	// Nothing changed: A stays invalid, B stays invalid, C stays active.
	eA, err := entryByRef(resA.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, eA.ValidUntil, "A should remain invalid")

	eB, err := entryByRef(resB.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, eB.ValidUntil, "B should remain invalid")

	// C, which superseded B, must remain active.
	eC, err := entryByRef(resC.Ref)
	require.NoError(t, err)
	assert.Empty(t, eC.ValidUntil, "C should remain active after A is restored")
}

func TestRestoreAlreadyActive(t *testing.T) {
	testMemoryDB(t)

	res, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "restore.alive",
		Predicate:          "is",
		Object:             "active",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	// Restore an active entry — should be a no-op
	result, err := Restore(RestoreInput{Refs: []string{res.Ref}})
	require.NoError(t, err)
	assert.Contains(t, result.RestoredRefs, res.Ref)
	assert.Empty(t, result.Conflicts)

	e, err := entryByRef(res.Ref)
	require.NoError(t, err)
	assert.Empty(t, e.ValidUntil)
}
