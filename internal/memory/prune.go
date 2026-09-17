package memory

import (
	"log"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// PruneSuperseded permanently deletes structured_memory rows that were
// invalidated (superseded by a newer fact, or explicitly forgotten) more than
// maxAgeDays ago. Superseded facts are soft-deleted via valid_until so
// recall can still reason about "what used to be true" for a while, but
// nothing ever purged old rows afterward — unlike contextnotes (a
// token/count quota, see limits.go) and the queries table (age-based
// retention, see db.RunQueryRetention), structured_memory grew unbounded.
func PruneSuperseded(maxAgeDays int) (int64, error) {
	if maxAgeDays <= 0 {
		maxAgeDays = 90
	}
	cutoff := time.Now().AddDate(0, 0, -maxAgeDays).Format("2006-01-02") + "T00:00:00"
	rows, err := db.ContextDB.Query(`SELECT ref FROM structured_memory WHERE valid_until IS NOT NULL AND valid_until != '' AND valid_until < ?`, cutoff)
	if err != nil {
		return 0, err
	}
	var refs []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" {
			refs = append(refs, ref)
		}
	}
	rows.Close()
	if len(refs) == 0 {
		return 0, nil
	}
	for _, ref := range refs {
		deleteFTS(ref)
		deleteMemoryVector(ref)
	}
	res, err := db.ContextDB.Exec(`DELETE FROM structured_memory WHERE valid_until IS NOT NULL AND valid_until != '' AND valid_until < ?`, cutoff)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		log.Printf("memory: pruned %d superseded fact(s)/procedure(s) older than %d days", n, maxAgeDays)
	}
	return n, nil
}

func deleteMemoryVector(ref string) {
	if conn, err := db.IndexReader(); err == nil {
		conn.Exec(`DELETE FROM vectors WHERE doc_type = 'memory' AND source_file = ?`, memoryVectorKey(ref))
	}
	search.Cache.DeleteNoteByRef(memoryVectorKey(ref))
}
