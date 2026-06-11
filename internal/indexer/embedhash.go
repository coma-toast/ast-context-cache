package indexer

import (
	"database/sql"

	"github.com/coma-toast/ast-context-cache/internal/search"
)

// ExpectedEmbedHash returns the content hash used for code vector embeddings.
func ExpectedEmbedHash(kind, name, filePath string, startLine, endLine int) string {
	fileCache := map[string][]string{}
	src := ReadSourceRange(filePath, startLine, endLine, fileCache)
	if len(src) > 500 {
		src = src[:500]
	}
	return search.ContentHash(kind + " " + name + ": " + src)
}

func deleteCodeVectorsTx(tx *sql.Tx, filePath, projectPath string) error {
	_, err := tx.Exec(`DELETE FROM vectors WHERE source_file = ? AND project_path = ? AND COALESCE(doc_type, 'code') = 'code'`,
		filePath, projectPath)
	return err
}
