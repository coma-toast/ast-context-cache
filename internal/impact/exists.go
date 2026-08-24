package impact

import (
	"encoding/json"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// Location is where a symbol is declared.
type Location struct {
	File string `json:"file"`
	Line int    `json:"line"`
	Kind string `json:"kind"`
}

// HandleCheckSymbolExists answers "is this name really declared anywhere?" — the
// cheap check before trusting a reference to a constant, method or test id that
// may have been renamed or deleted. Sibling checkouts of the same repo are
// searched too, so a declaration that only exists on another branch is visible.
func HandleCheckSymbolExists(args map[string]interface{}, projectPath string) string {
	symbol := strArg(args, "symbol")
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	if symbol == "" {
		return `{"error": "symbol required"}`
	}

	scopeFrag, scopeArgs, scope := projectlinks.ScopeSQLWithRepoSiblings("", projectPath, true)
	conn, err := db.IndexReader()
	if err != nil {
		return errJSON(err)
	}
	rows, err := conn.Query(
		"SELECT file, COALESCE(start_line,0), kind FROM symbols WHERE "+scopeFrag+" AND LOWER(name) = ? ORDER BY file, start_line",
		append(scopeArgs, strings.ToLower(symbol))...)
	if err != nil {
		return errJSON(err)
	}
	defer rows.Close()

	locations := []Location{}
	for rows.Next() {
		var loc Location
		if rows.Scan(&loc.File, &loc.Line, &loc.Kind) != nil {
			continue
		}
		loc.File = RelPathInScope(loc.File, projectPath, scope)
		locations = append(locations, loc)
	}

	data, _ := json.Marshal(map[string]interface{}{
		"symbol":        symbol,
		"exists":        len(locations) > 0,
		"locations":     locations,
		"checked_scope": scope,
	})
	return string(data)
}
