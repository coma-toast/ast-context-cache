package indexer

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const plaintextMaxBytes = 2 << 20 // 2 MiB per file for FTS

func indexLogFilesEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(db.GetSetting("index_log_files", "false")))
	return v == "true" || v == "1" || v == "yes"
}

// ShouldSkipEmbed is true for plaintext log/text blobs (search-only, no vectors).
func ShouldSkipEmbed(path string) bool {
	return GetLanguage(path) == "plaintext"
}

// indexPlaintextFile used to open its own transaction directly on the reader
// pool (conn.Begin()) instead of going through db.IndexWrite, the
// single-writer queue every other indexing path (and the WAL-checkpoint
// drain) relies on to serialize writes. Now uses db.IndexWrite like IndexFile.
func indexPlaintextFile(filePath, projectPath string) (count, fullTokens, skeletonTokens int, err error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, 0, 0, err
	}
	if len(content) > plaintextMaxBytes {
		content = content[:plaintextMaxBytes]
	}
	text := string(content)
	lines := strings.Split(text, "\n")
	nLines := len(lines)
	first := ""
	if nLines > 0 {
		first = strings.TrimSpace(lines[0])
	}
	name := filepath.Base(filePath)
	fqn := fmt.Sprintf("%s#plaintext", filePath)
	fullTokens = db.EstimateTokens(text)
	skeletonTokens = db.EstimateTokens(first)
	embedHash := ExpectedEmbedHash("plaintext", name, filePath, 1, nLines)

	err = db.IndexWrite(func(tx *sql.Tx) error {
		if err := deleteCodeVectorsTx(tx, filePath, projectPath); err != nil {
			return err
		}
		if _, err := tx.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath); err != nil {
			return err
		}
		if _, err := tx.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath); err != nil {
			return err
		}
		if _, err := tx.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, skeleton, embed_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			name, "plaintext", filePath, 1, nLines, text, fqn, projectPath, first, embedHash); err != nil {
			return err
		}
		return db.UpsertIndexedFileWith(tx, filePath, projectPath, time.Now())
	})
	if err != nil {
		return 0, fullTokens, skeletonTokens, err
	}
	search.Cache.DeleteByFile(filePath, projectPath)
	notifyIndexCommitted()
	return 1, fullTokens, skeletonTokens, nil
}
