package impact

import (
	"encoding/json"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// Location is where a symbol is declared. QualifiedName is set for a member
// (e.g. LlamaCppClient.load_model) so same-named methods can be told apart.
type Location struct {
	File          string `json:"file"`
	Line          int    `json:"line"`
	Kind          string `json:"kind"`
	QualifiedName string `json:"qualified_name,omitempty"`
}

// NameMatchSQL returns a WHERE fragment (over alias's name/fqn columns) matching
// symbol case-insensitively by name. A dotted symbol such as
// "LlamaCppClient.load_model" also matches a member whose qualified name ends
// with it, since the indexer stores members as name=load_model,
// fqn=<file>.LlamaCppClient.load_model.
func NameMatchSQL(alias, symbol string) (string, []interface{}) {
	return nameMatchSQL(alias, symbol, true)
}

// exactNameMatchSQL is NameMatchSQL without case folding, for callers where
// HOSTS and a YAML key hosts must stay different symbols.
func exactNameMatchSQL(alias, symbol string) (string, []interface{}) {
	return nameMatchSQL(alias, symbol, false)
}

func nameMatchSQL(alias, symbol string, fold bool) (string, []interface{}) {
	col := func(c string) string {
		if alias != "" {
			c = alias + "." + c
		}
		if fold {
			return "LOWER(" + c + ")"
		}
		return c
	}
	if fold {
		symbol = strings.ToLower(symbol)
	}
	i := strings.LastIndex(symbol, ".")
	if i <= 0 || i == len(symbol)-1 {
		return col("name") + " = ?", []interface{}{symbol}
	}
	suffix := "." + symbol
	return "(" + col("name") + " = ? OR (" + col("name") + " = ? AND (" + col("fqn") + " = ? OR SUBSTR(" + col("fqn") + ", -LENGTH(?)) = ?)))",
		[]interface{}{symbol, symbol[i+1:], symbol, suffix, suffix}
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
	nameFrag, nameArgs := NameMatchSQL("", symbol)
	rows, err := conn.Query(
		"SELECT file, COALESCE(start_line,0), kind, name, COALESCE(fqn,'') FROM symbols WHERE "+scopeFrag+" AND "+nameFrag+" ORDER BY file, start_line",
		append(scopeArgs, nameArgs...)...)
	if err != nil {
		return errJSON(err)
	}
	defer rows.Close()

	locations := []Location{}
	for rows.Next() {
		var loc Location
		var name, fqn string
		if rows.Scan(&loc.File, &loc.Line, &loc.Kind, &name, &fqn) != nil {
			continue
		}
		if q := db.QualifiedName(fqn, loc.File, name); q != name {
			loc.QualifiedName = q
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
