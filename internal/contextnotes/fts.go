package contextnotes

import "github.com/coma-toast/ast-context-cache/internal/db"

func indexNoteFTS(ref, sessionID, label, content string) {
	db.DB.Exec(`INSERT INTO context_notes_fts (ref, session_id, label, content) VALUES (?, ?, ?, ?)`,
		ref, sessionID, label, content)
}

func deleteNoteFTS(ref string) {
	db.DB.Exec(`DELETE FROM context_notes_fts WHERE ref = ?`, ref)
}
