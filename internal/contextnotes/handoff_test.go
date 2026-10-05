package contextnotes

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func storeNote(t *testing.T, sid, content, kind string) string {
	t.Helper()
	res, err := Store(sid, content, "lbl", "/p", nil, kind, nil, nil)
	require.NoError(t, err)
	return res.Ref
}

func addHandoffChild(t *testing.T, sid string) {
	t.Helper()
	_, err := db.ContextDB.Exec(`INSERT INTO handoff_children (child_session_id, handoff_ref, tree_id) VALUES (?, 'hof_x', 'hft_x')`, sid)
	require.NoError(t, err)
}

func TestPeekRecordsNoAccess(t *testing.T) {
	testNotesDB(t)
	ref := storeNote(t, "peek-s", "peek content", "")
	n, err := Peek(ref)
	require.NoError(t, err)
	assert.Equal(t, "peek content", n.Content)
	assert.Equal(t, "peek-s", n.SessionID)
	var access int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT access_count FROM context_notes WHERE ref = ?`, ref).Scan(&access))
	assert.Zero(t, access)
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM context_note_access WHERE ref = ?`, ref).Scan(&access))
	assert.Zero(t, access)
	_, err = Peek("ctx_missing")
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
	_, err = Peek(" ")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
}

func TestFlushSessionDeletesNotesFTSAndStats(t *testing.T) {
	testNotesDB(t)
	a := storeNote(t, "flush-s", "first child note", "")
	b := storeNote(t, "flush-s", "second child note", KindHandoffResult)
	other := storeNote(t, "flush-other", "unrelated note", "")
	n, err := FlushSession("flush-s")
	require.NoError(t, err)
	assert.Equal(t, 2, n)
	assert.False(t, noteExists(t, a))
	assert.False(t, noteExists(t, b))
	assert.True(t, noteExists(t, other))
	var fts int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_notes_fts WHERE session_id = 'flush-s'`).Scan(&fts))
	assert.Zero(t, fts)
	r := SessionRollupFor("flush-s")
	assert.Zero(t, r.NotesCount)
	assert.Zero(t, r.VirtualTokensStored)
	n, err = FlushSession("flush-s")
	require.NoError(t, err)
	assert.Zero(t, n)
}

func TestFlushOrphansSkipsTreeOwnedNotes(t *testing.T) {
	testNotesDB(t)
	addHandoffChild(t, "child-s")
	result := storeNote(t, "plain-s", "a handoff result", KindHandoffResult)
	childNote := storeNote(t, "child-s", "child scratch work", "")
	orphan := storeNote(t, "plain-s", "an ordinary orphan", "")
	_, err := db.ContextDB.Exec(`UPDATE context_notes SET created_at = datetime('now', '-2 hours')`)
	require.NoError(t, err)
	res, _, err := FlushOrphans("", OrphanPurgeGrace)
	require.NoError(t, err)
	assert.Equal(t, 1, res.FlushedRefs)
	assert.False(t, noteExists(t, orphan))
	assert.True(t, noteExists(t, result), "handoff results expire with their tree")
	assert.True(t, noteExists(t, childNote), "child-session notes expire with their tree")
}

func TestLRUEvictionSkipsTreeOwnedNotes(t *testing.T) {
	testNotesDB(t)
	require.NoError(t, db.SetSetting("context_max_notes_session", "2"))
	require.NoError(t, db.SetSetting("context_limit_policy", "lru_session"))
	result := storeNote(t, "lru-h", "result note", KindHandoffResult)
	plain := storeNote(t, "lru-h", "plain note", "")
	res, err := Store("lru-h", "third note", "", "", nil, "", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{plain}, res.EvictedRefs, "the result is skipped, the next oldest goes")
	assert.True(t, noteExists(t, result))

	addHandoffChild(t, "lru-child")
	storeNote(t, "lru-child", "child one", "")
	storeNote(t, "lru-child", "child two", "")
	_, err = Store("lru-child", "child three", "", "", nil, "", nil, nil)
	require.Error(t, err, "a child session's notes are never evicted, so the cap rejects")
	assert.True(t, errs.HasCode(err, errs.CodeLimitExceeded), "%v", err)
}
