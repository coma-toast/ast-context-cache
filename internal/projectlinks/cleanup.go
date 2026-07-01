package projectlinks

import (
	"log"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

var onLinkCleanup func(parent, child string)

// SetOnLinkCleanup registers a callback after DB duplicate purge (e.g. embed queue cleanup).
func SetOnLinkCleanup(fn func(parent, child string)) {
	onLinkCleanup = fn
}

// CleanupParentDuplicates removes parent-owned index rows for files under child.
func CleanupParentDuplicates(parent, child string) error {
	parent = NormalizePath(parent)
	child = NormalizePath(child)
	if parent == "" || child == "" || db.IndexDB == nil {
		return nil
	}
	prefix := child
	if !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}
	like := prefix + "%"
	if err := cleanupParentDuplicates(parent, child, like); err != nil {
		return err
	}
	if onLinkCleanup != nil {
		onLinkCleanup(parent, child)
	}
	return nil
}

func cleanupParentDuplicates(parent, child, like string) error {
	if db.IndexDB == nil {
		return nil
	}
	queries := []string{
		`DELETE FROM symbols WHERE project_path = ? AND (file = ? OR file LIKE ?)`,
		`DELETE FROM edges WHERE project_path = ? AND (source_file = ? OR source_file LIKE ?)`,
		`DELETE FROM vectors WHERE project_path = ? AND (source_file = ? OR source_file LIKE ?)`,
		`DELETE FROM indexed_files WHERE project_path = ? AND (file = ? OR file LIKE ?)`,
		`DELETE FROM summaries WHERE project_path = ? AND (file_path = ? OR file_path LIKE ?)`,
		`DELETE FROM embed_pending WHERE project_path = ? AND (file = ? OR file LIKE ?)`,
	}
	var total int64
	for _, q := range queries {
		res, err := db.IndexDB.Exec(q, parent, child, like)
		if err != nil {
			log.Printf("projectlinks: cleanup: %v", err)
			continue
		}
		if n, err := res.RowsAffected(); err == nil {
			total += n
		}
	}
	if total > 0 {
		log.Printf("projectlinks: removed %d duplicate rows for parent=%s child=%s", total, parent, child)
	}
	return nil
}

// LinkStats returns symbol and file counts for a project path.
func LinkStats(projectPath string) (symbols, files int) {
	projectPath = NormalizePath(projectPath)
	if projectPath == "" || db.IndexDB == nil {
		return 0, 0
	}
	db.IndexDB.QueryRow(`SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?`, projectPath).Scan(&symbols, &files)
	return symbols, files
}
