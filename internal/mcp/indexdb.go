package mcp

import (
	"database/sql"
	"encoding/json"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func indexDBErrJSON(err error) string {
	data, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(data)
}

func indexDBOrErr() (*sql.DB, error) {
	return db.IndexReader()
}
