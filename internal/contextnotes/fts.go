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
