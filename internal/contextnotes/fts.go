package contextnotes

import "github.com/coma-toast/ast-context-cache/internal/db"

const (
	insertNoteFTSQuery = `INSERT INTO context_notes_fts (ref, session_id, label, content) VALUES (?, ?, ?, ?)`
	deleteNoteFTSQuery = `DELETE FROM context_notes_fts WHERE ref = ?`
)

func indexNoteFTS(ref, sessionID, label, content string) {
	db.ContextDB.Exec(insertNoteFTSQuery,
		ref, sessionID, label, content)
}

func deleteNoteFTS(ref string) {
	db.ContextDB.Exec(deleteNoteFTSQuery, ref)
}

// reindexNoteFTS replaces a note's FTS row after an in-place content edit.
// context_notes_fts is a bare fts5 table with no triggers, so an edit that only
// updated context_notes would leave search matching the old body.
func reindexNoteFTS(ref, sessionID, label, content string) {
	deleteNoteFTS(ref)
	indexNoteFTS(ref, sessionID, label, content)
}
