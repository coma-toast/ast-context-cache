package contextnotes

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	testInsertTombstoneQuery = `INSERT INTO context_note_tombstones (ref, kind, tool, args_json, expired_at) VALUES (?, 'offload', 't', '', ?)`
	testCountTombstoneQuery  = `SELECT COUNT(*) FROM context_note_tombstones WHERE ref = ?`
)

func TestOffloadStubFormat(t *testing.T) {
	assert.Equal(t, "[ctx_abc get_context_capsule auth flow, 9.4k tok — head shown, fetch_context for all]",
		OffloadStub("ctx_abc", "get_context_capsule", "auth flow", 9400))
	assert.Equal(t, "[ctx_abc get_file_context a.go, 812 tok — head shown, fetch_context for all]",
		OffloadStub("ctx_abc", "get_file_context", "a.go", 812))
	assert.Equal(t, "[ctx_abc retrieve, 12k tok — head shown, fetch_context for all]",
		OffloadStub("ctx_abc", "retrieve", "", 12000))
}

func TestOffloadExcludedFromQuota(t *testing.T) {
	testNotesDB(t)
	require.NoError(t, db.SetSetting("context_max_notes_session", "1"))
	require.NoError(t, db.SetSetting("context_max_tokens_session", "200"))
	big, err := StoreOffload("sess-q", "/p", strings.Repeat("word ", 400), "retrieve", map[string]any{"query": "q"}, "q")
	require.NoError(t, err)
	assert.Equal(t, "retrieve q", big.Label)
	_, err = Store("sess-q", "normal note", "n", "/p", nil, "", nil, nil)
	require.NoError(t, err)
	sn, st, gn, _ := QuotaForSession("sess-q")
	assert.Equal(t, 1, sn)
	assert.Equal(t, 1, gn)
	assert.Less(t, st, big.VirtualTokensStored)
	note, err := noteByRef(big.Ref)
	require.NoError(t, err)
	assert.Equal(t, KindOffload, note.Kind)
	assert.JSONEq(t, `{"tool":"retrieve","args":{"query":"q"}}`, note.MetadataJSON)
}

func TestOffloadLimitEvictsOldest(t *testing.T) {
	testNotesDB(t)
	require.NoError(t, db.SetSetting("context_offload_max_tokens_global", "300"))
	text := strings.Repeat("word ", 120)
	first, err := StoreOffload("sess-o", "", text, "get_file_context", map[string]any{"file": "a.go"}, "a.go")
	require.NoError(t, err)
	_, err = StoreOffload("sess-o", "", text, "get_file_context", nil, "b.go")
	require.NoError(t, err)
	third, err := StoreOffload("sess-o", "", text, "get_file_context", nil, "c.go")
	require.NoError(t, err)
	assert.Equal(t, []string{first.Ref}, third.EvictedRefs)
	fetch, err := Fetch([]string{first.Ref}, "", "")
	require.NoError(t, err)
	assert.Empty(t, fetch.Notes)
	require.Len(t, fetch.Expired, 1)
	assert.Equal(t, ExpiredRef{Ref: first.Ref, Status: ExpiredStatus, Tool: "get_file_context", Args: map[string]any{"file": "a.go"}}, fetch.Expired[0])
	_, err = StoreOffload("sess-o", "", strings.Repeat("word ", 400), "retrieve", nil, "x")
	var le *LimitError
	require.ErrorAs(t, err, &le)
	assert.Equal(t, "offload_tokens", le.Limit)
}

func TestOffloadExpiryTombstone(t *testing.T) {
	testNotesDB(t)
	off, err := StoreOffload("sess-p", "", "full offloaded result", "get_context_capsule", map[string]any{"query": "auth"}, "auth")
	require.NoError(t, err)
	fresh, err := StoreOffload("sess-p", "", "fresh offloaded result", "retrieve", nil, "x")
	require.NoError(t, err)
	plain, err := Store("sess-p", "plain note", "p", "", nil, "", nil, nil)
	require.NoError(t, err)
	for _, ref := range []string{off.Ref, plain.Ref} {
		_, err = db.ContextDB.Exec(testBackdateNoteQuery, "-48 hours", ref)
		require.NoError(t, err)
	}
	_, err = db.ContextDB.Exec(testInsertTombstoneQuery, "ctx_old", db.SQLTime(time.Now().Add(-40*24*time.Hour)))
	require.NoError(t, err)
	n, err := PurgeExpiredOffloads(24 * time.Hour)
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	fetch, err := Fetch([]string{off.Ref, fresh.Ref, plain.Ref}, "sess-p", "")
	require.NoError(t, err)
	assert.Len(t, fetch.Notes, 2)
	require.Len(t, fetch.Expired, 1)
	assert.Equal(t, ExpiredRef{Ref: off.Ref, Status: ExpiredStatus, Tool: "get_context_capsule", Args: map[string]any{"query": "auth"}}, fetch.Expired[0])
	var old int
	require.NoError(t, db.ContextDB.QueryRow(testCountTombstoneQuery, "ctx_old").Scan(&old))
	assert.Zero(t, old)
}
