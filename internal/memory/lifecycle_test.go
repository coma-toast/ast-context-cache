package memory

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestRunExpiryJob(t *testing.T) {
	testMemoryDB(t)

	past := time.Now().Add(-2 * time.Hour).Format(time.RFC3339)
	res1, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "test.ttl", Predicate: "is", Object: "expired",
		ExpiresAt: past,
	})
	require.NoError(t, err)

	res2, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "test.ttl", Predicate: "is", Object: "active",
	})
	require.NoError(t, err)

	RunExpiryJob()

	e1, err := entryByRef(res1.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, e1.ValidUntil)
	assert.Equal(t, ReasonTTL, e1.InvalidatedReason)

	e2, err := entryByRef(res2.Ref)
	require.NoError(t, err)
	assert.Empty(t, e2.ValidUntil)
}

func TestPruneArchivedDaily(t *testing.T) {
	testMemoryDB(t)

	resOld, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "archive", Predicate: "is", Object: "old",
	})
	require.NoError(t, err)

	resNew, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "archive", Predicate: "is", Object: "new",
	})
	require.NoError(t, err)

	oldTime := time.Now().AddDate(0, 0, -3).Format("2006-01-02 15:04:05")
	newTime := time.Now().Format("2006-01-02 15:04:05")

	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ?, invalidated_reason = ? WHERE ref = ?`, oldTime, ReasonUserForget, resOld.Ref)
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ?, invalidated_reason = ? WHERE ref = ?`, newTime, ReasonUserForget, resNew.Ref)
	require.NoError(t, err)

	n, err := PruneArchived(1)
	require.NoError(t, err)
	assert.Equal(t, int64(1), n)

	assertGone(t, resOld.Ref)
	assertPresent(t, resNew.Ref)
}

func TestRunStaleAnchorJob(t *testing.T) {
	testMemoryDB(t)

	past := time.Now().Add(-2 * time.Hour).Format(time.RFC3339)
	res, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "anchor", Predicate: "is", Object: "stale",
		WorldInvalidAt: past,
	})
	require.NoError(t, err)

	RunStaleAnchorJob()

	e, err := entryByRef(res.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, e.ValidUntil)
	assert.Equal(t, ReasonStaleAnchor, e.InvalidatedReason)
}

func TestRejectQuarantined(t *testing.T) {
	testMemoryDB(t)

	res, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "quarantine", Predicate: "is", Object: "bad",
	})
	require.NoError(t, err)

	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET quarantined = 1 WHERE ref = ?`, res.Ref)
	require.NoError(t, err)

	err = RejectQuarantined(res.Ref)
	require.NoError(t, err)

	e, err := entryByRef(res.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, e.ValidUntil)
	assert.Equal(t, ReasonQuarantine, e.InvalidatedReason)
}
