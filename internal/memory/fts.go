package memory

import "github.com/coma-toast/ast-context-cache/internal/db"

const (
	insertEntryFTSQuery = `INSERT INTO structured_memory_fts (ref, subject, predicate, object, rule) VALUES (?, ?, ?, ?, ?)`
	deleteEntryFTSQuery = `DELETE FROM structured_memory_fts WHERE ref = ?`
)

func indexFTS(ref, subject, predicate, object, rule string) {
	db.ContextDB.Exec(insertEntryFTSQuery,
		ref, subject, predicate, object, rule)
}

func deleteFTS(ref string) {
	db.ContextDB.Exec(deleteEntryFTSQuery, ref)
}
