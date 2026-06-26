package context

import (
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// SymbolDedupKey identifies a symbol within a session.
func SymbolDedupKey(file, name string, startLine int) string {
	return file + "|" + name + "|" + strconv.Itoa(startLine)
}

func GetReturnedSymbolKeys(sessionID string) map[string]bool {
	if sessionID == "" {
		return nil
	}
	rows, err := db.DB.Query(`
		SELECT COALESCE(file_path,''), COALESCE(symbol_name,''), COALESCE(start_line,0)
		FROM sessions
		WHERE session_id = ? AND (symbol_name != '' OR file_path != '')`, sessionID)
	if err != nil {
		return nil
	}
	defer rows.Close()
	seen := map[string]bool{}
	for rows.Next() {
		var file, name string
		var startLine int
		rows.Scan(&file, &name, &startLine)
		if file != "" && name != "" {
			seen[SymbolDedupKey(file, name, startLine)] = true
		}
	}
	return seen
}

func LookupSymbolID(file, name, projectPath string, startLine int) int {
	var id int
	err := db.IndexDB.QueryRow(
		"SELECT id FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
		file, name, projectPath, startLine).Scan(&id)
	if err != nil {
		return 0
	}
	return id
}

func LogReturned(sessionID, file, name, projectPath string, startLine int, mode string, tokenCount int) {
	if sessionID == "" {
		return
	}
	symbolID := LookupSymbolID(file, name, projectPath, startLine)
	db.EnqueueSessionReturned(sessionID, symbolID, name, startLine, file, mode, tokenCount)
}
