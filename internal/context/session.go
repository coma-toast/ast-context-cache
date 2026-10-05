package context

import (
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	selectReturnedSymbolsQuery = `
		SELECT COALESCE(file_path,''), COALESCE(symbol_name,''), COALESCE(start_line,0)
		FROM sessions
		WHERE session_id = ? AND (symbol_name != '' OR file_path != '')`
	selectSymbolIDQuery = "SELECT id FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1"
)

// SymbolDedupKey identifies a symbol within a session.
func SymbolDedupKey(file, name string, startLine int) string {
	return file + "|" + name + "|" + strconv.Itoa(startLine)
}

// loadReturnedKeys reads the dedup keys persisted for sessionID. Rows still in the
// write buffer are not visible here; the session store covers them in memory.
func loadReturnedKeys(sessionID string) (map[string]struct{}, error) {
	rows, err := db.DB.Query(selectReturnedSymbolsQuery, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	seen := map[string]struct{}{}
	for rows.Next() {
		var file, name string
		var startLine int
		if err := rows.Scan(&file, &name, &startLine); err != nil {
			return nil, err
		}
		if file != "" && name != "" {
			seen[SymbolDedupKey(file, name, startLine)] = struct{}{}
		}
	}
	return seen, rows.Err()
}

// LookupSymbolID returns the index row id of a symbol, or 0 when it is not indexed.
func LookupSymbolID(file, name, projectPath string, startLine int) int {
	var id int
	conn, err := db.IndexReader()
	if err != nil {
		return 0
	}
	err = conn.QueryRow(selectSymbolIDQuery, file, name, projectPath, startLine).Scan(&id)
	if err != nil {
		return 0
	}
	return id
}

// LogReturned records that one symbol was delivered to sessionID; see MarkReturned.
func LogReturned(sessionID, file, name, projectPath string, startLine int, mode string, tokenCount int) {
	MarkReturned(sessionID, ReturnedSymbol{File: file, Name: name, ProjectPath: projectPath, StartLine: startLine, Mode: mode, Tokens: tokenCount})
}
