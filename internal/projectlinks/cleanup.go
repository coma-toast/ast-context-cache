package projectlinks

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	deleteParentDupSymbolsQuery      = `DELETE FROM symbols WHERE project_path = ? AND (file = ? OR file LIKE ?)`
	deleteParentDupEdgesQuery        = `DELETE FROM edges WHERE project_path = ? AND (source_file = ? OR source_file LIKE ?)`
	deleteParentDupVectorsQuery      = `DELETE FROM vectors WHERE project_path = ? AND (source_file = ? OR source_file LIKE ?)`
	deleteParentDupIndexedFilesQuery = `DELETE FROM indexed_files WHERE project_path = ? AND (file = ? OR file LIKE ?)`
	deleteParentDupSummariesQuery    = `DELETE FROM summaries WHERE project_path = ? AND (file_path = ? OR file_path LIKE ?)`
	deleteParentDupEmbedPendingQuery = `DELETE FROM embed_pending WHERE project_path = ? AND (file = ? OR file LIKE ?)`
	countProjectSymbolsQuery         = `SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?`
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
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	queries := []string{
		deleteParentDupSymbolsQuery,
		deleteParentDupEdgesQuery,
		deleteParentDupVectorsQuery,
		deleteParentDupIndexedFilesQuery,
		deleteParentDupSummariesQuery,
		deleteParentDupEmbedPendingQuery,
	}
	var total int64
	for _, q := range queries {
		res, err := conn.Exec(q, parent, child, like)
		if err != nil {
			logger.Warn("Failed to clean up parent duplicate rows", "parent", parent, "child", child, "error", err)
			continue
		}
		if n, err := res.RowsAffected(); err == nil {
			total += n
		}
	}
	if total > 0 {
		logger.Info("Removed parent duplicate rows", "rows", total, "parent", parent, "child", child)
	}
	return nil
}

// LinkStats returns symbol and file counts for a project path.
func LinkStats(projectPath string) (symbols, files int) {
	projectPath = NormalizePath(projectPath)
	if projectPath == "" {
		return 0, 0
	}
	conn, err := db.IndexReader()
	if err != nil {
		return 0, 0
	}
	conn.QueryRow(countProjectSymbolsQuery, projectPath).Scan(&symbols, &files)
	return symbols, files
}
