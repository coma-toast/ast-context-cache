package memory

import "github.com/coma-toast/ast-context-cache/internal/db"

func indexFTS(ref, subject, predicate, object, rule string) {
	db.DB.Exec(`INSERT INTO structured_memory_fts (ref, subject, predicate, object, rule) VALUES (?, ?, ?, ?, ?)`,
		ref, subject, predicate, object, rule)
}

func deleteFTS(ref string) {
	db.DB.Exec(`DELETE FROM structured_memory_fts WHERE ref = ?`, ref)
}
