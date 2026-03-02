package context

import (
	"github.com/coma-toast/ast-context-cache/internal/db"
)

func GetReturnedFiles(sessionID string) map[string]bool {
	if sessionID == "" {
		return nil
	}
	rows, err := db.DB.Query("SELECT DISTINCT file_path FROM sessions WHERE session_id = ?", sessionID)
	if err != nil {
		return nil
	}
	defer rows.Close()

	seen := map[string]bool{}
	for rows.Next() {
		var f string
		rows.Scan(&f)
		seen[f] = true
	}
	return seen
}

func GetReturnedSymbols(sessionID string) map[string]bool {
	if sessionID == "" {
		return nil
	}
	rows, err := db.DB.Query("SELECT file_path, COALESCE(mode,'') FROM sessions WHERE session_id = ?", sessionID)
	if err != nil {
		return nil
	}
	defer rows.Close()

	seen := map[string]bool{}
	for rows.Next() {
		var f, m string
		rows.Scan(&f, &m)
		seen[f+"|"+m] = true
	}
	return seen
}

func LogReturned(sessionID string, file string, symbolID int, mode string, tokenCount int) {
	if sessionID == "" {
		return
	}
	db.DB.Exec("INSERT INTO sessions (session_id, symbol_id, file_path, mode, token_count) VALUES (?, ?, ?, ?, ?)",
		sessionID, symbolID, file, mode, tokenCount)
}
