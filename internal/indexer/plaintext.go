package indexer

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
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

func indexPlaintextFile(filePath, projectPath string) (count, fullTokens, skeletonTokens int, err error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, 0, 0, err
	}
	if len(content) > plaintextMaxBytes {
		content = content[:plaintextMaxBytes]
	}
	db.DB.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath)
	db.DB.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath)
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
	_, err = db.DB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, skeleton) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		name, "plaintext", filePath, 1, nLines, text, fqn, projectPath, first)
	if err != nil {
		return 0, fullTokens, skeletonTokens, err
	}
	db.UpsertIndexedFile(filePath, projectPath, time.Now())
	return 1, fullTokens, skeletonTokens, nil
}
