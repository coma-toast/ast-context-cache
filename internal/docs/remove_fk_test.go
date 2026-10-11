package docs

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

const (
	testForeignKeysPragma         = `PRAGMA foreign_keys`
	testCountSourceByIDQuery      = `SELECT COUNT(*) FROM doc_sources WHERE id = ?`
	testCountContentBySourceQuery = `SELECT COUNT(*) FROM doc_content WHERE source_id = ?`
)

// doc_content.source_id references doc_sources(id), and the pools enforce foreign keys,
// so removing or refreshing a source must delete its sections before the source row.
func TestRemoveSourceWithForeignKeysEnforced(t *testing.T) {
	dbtest.Init(t)
	var fk int
	require.NoError(t, db.ContextDB.QueryRow(testForeignKeysPragma).Scan(&fk))
	require.Equal(t, 1, fk)
	id, err := AddSource("fk-docs", "markdown", "file:///nowhere.md", "")
	require.NoError(t, err)
	require.NoError(t, storeEntries(id, []DocEntry{{Title: "Intro", Content: "hello"}, {Title: "More", Content: "world"}}))
	_, err = db.ContextDB.Exec(deleteDocSourceQuery, id)
	require.Error(t, err, "deleting a source that still has sections must violate the foreign key")

	require.NoError(t, RemoveSource(id))
	var sources, sections int
	require.NoError(t, db.ContextDB.QueryRow(testCountSourceByIDQuery, id).Scan(&sources))
	require.NoError(t, db.ContextDB.QueryRow(testCountContentBySourceQuery, id).Scan(&sections))
	assert.Zero(t, sources)
	assert.Zero(t, sections)
}
