package contextnotes

import (
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// TestStorePinnedRejectsOverCap covers AC19: a pinned note over the 500-token cap
// is rejected and the 500 cap is stated in the error (PIN-1).
func TestStorePinnedRejectsOverCap(t *testing.T) {
	testNotesDB(t)
	long := strings.Repeat("word ", 700)
	_, err := StorePinned("/tmp/proj-pinned", long, "user")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max 500")
}

// TestStorePinnedOnePerProject covers PIN-1: storing twice keeps one note with the
// same ref and the updated content.
func TestStorePinnedOnePerProject(t *testing.T) {
	testNotesDB(t)
	first, err := StorePinned("/tmp/proj-pinned", "first note", "user")
	require.NoError(t, err)

	second, err := StorePinned("/tmp/proj-pinned", "second note", "user")
	require.NoError(t, err)
	assert.Equal(t, first.Ref, second.Ref, "ref must be kept across updates")
	assert.Equal(t, "second note", second.Content)

	// Pinned returns the latest content (PIN-1 read side).
	got, err := Pinned("/tmp/proj-pinned")
	require.NoError(t, err)
	assert.Equal(t, "second note", got.Content)
	assert.Equal(t, first.Ref, got.Ref)
}

// TestStorePinnedKeepsRevisionRow verifies PIN-1: updating a pinned note writes a
// revision row into context_note_revisions and keeps the ref stable.
func TestStorePinnedKeepsRevisionRow(t *testing.T) {
	testNotesDB(t)
	first, err := StorePinned("/tmp/proj-revision", "first note", "user")
	require.NoError(t, err)

	var revCount int
	err = db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_note_revisions WHERE ref = ?`, first.Ref).Scan(&revCount)
	require.NoError(t, err)
	require.Equal(t, 0, revCount, "no revisions before the second store")

	_, err = StorePinned("/tmp/proj-revision", "second note", "user")
	require.NoError(t, err)

	// Check a revision was written.
	revCount = 0
	err = db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_note_revisions WHERE ref = ?`, first.Ref).Scan(&revCount)
	require.NoError(t, err)
	require.Equal(t, 1, revCount, "one revision row should exist after second store")

	// Content of the revision should be the old note content
	var revContent string
	err = db.ContextDB.QueryRow(`SELECT content FROM context_note_revisions WHERE ref = ? AND revision = 1`, first.Ref).Scan(&revContent)
	require.NoError(t, err)
	assert.Equal(t, "first note", revContent)

	// Latest pinned content is updated
	got, err := Pinned("/tmp/proj-revision")
	require.NoError(t, err)
	assert.Equal(t, "second note", got.Content)
	assert.Equal(t, first.Ref, got.Ref)
}

// TestPinnedMissingProject errs on an empty project path.
func TestPinnedMissingProject(t *testing.T) {
	testNotesDB(t)
	_, err := Pinned("")
	require.Error(t, err)
	assert.Equal(t, errs.CodeInvalidInput, errs.CodeOf(err))
}

// TestRoutePinnedKind covers store_context with kind=pinned routing to StorePinned.
func TestRoutePinnedKind(t *testing.T) {
	testNotesDB(t)
	note, err := routePinnedKind("/tmp/proj-route", "routed content")
	require.NoError(t, err)
	assert.Equal(t, "routed content", note.Content)

	got, err := Pinned("/tmp/proj-route")
	require.NoError(t, err)
	assert.Equal(t, note.Ref, got.Ref)
}
