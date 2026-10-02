package mcp

import (
	"database/sql"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/impact"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// handleCacheSummary stores an LLM-written summary for an indexed symbol (or,
// with no symbol, for an indexed file). It refuses names the index doesn't
// know: summary mode looks summaries up by an indexed symbol's qualified name
// (Class.method for a member, so same-named methods of different classes keep
// separate summaries), and a row keyed by anything else would be reported
// "cached" yet never be served.
func handleCacheSummary(args map[string]interface{}, projectPath string) map[string]interface{} {
	file, _ := args["file"].(string)
	summary, _ := args["summary"].(string)
	symbol, _ := args["symbol"].(string)
	symbol = strings.TrimSpace(symbol)
	if file == "" || summary == "" || projectPath == "" {
		return map[string]interface{}{"error": "file, summary, and project_path required"}
	}
	if !filepath.IsAbs(file) {
		file = filepath.Join(projectPath, file)
	}
	file = filepath.Clean(file)
	conn, err := indexDBOrErr()
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	owner := projectlinks.OwningProject(file, projectPath)
	rel := db.RelPath(file, projectPath)

	name, qualified, contentHash := "", "", ""
	if symbol == "" {
		var one int
		if conn.QueryRow(`SELECT 1 FROM indexed_files WHERE file = ? AND project_path = ?`, file, owner).Scan(&one) != nil &&
			conn.QueryRow(`SELECT 1 FROM symbols WHERE file = ? AND project_path = ? LIMIT 1`, file, owner).Scan(&one) != nil {
			return map[string]interface{}{
				"error": fmt.Sprintf("file %s is not indexed for this project; index it with index_files before caching a summary", rel),
				"file":  file,
			}
		}
	} else {
		nameFrag, nameArgs := impact.NameMatchSQL("", symbol)
		rows, err := conn.Query(
			`SELECT name, COALESCE(code,''), COALESCE(fqn,'') FROM symbols WHERE file = ? AND project_path = ? AND `+nameFrag+` ORDER BY start_line`,
			append([]interface{}{file, owner}, nameArgs...)...)
		if err != nil {
			return map[string]interface{}{"error": err.Error()}
		}
		var code string
		var candidates []string
		for rows.Next() {
			var n, c, fqn string
			if rows.Scan(&n, &c, &fqn) != nil {
				continue
			}
			q := db.QualifiedName(fqn, file, n)
			if len(candidates) == 0 {
				name, code, qualified = n, c, q
			}
			// A bare name shared by several classes' methods can't pick one.
			if !slices.Contains(candidates, q) {
				candidates = append(candidates, q)
			}
		}
		rows.Close()
		if len(candidates) == 0 {
			return map[string]interface{}{
				"error":        fmt.Sprintf("symbol %q is not indexed in %s; nothing was cached (check the name with get_file_context or check_symbol_exists)", symbol, rel),
				"file":         file,
				"symbol":       symbol,
				"did_you_mean": similarSymbols(conn, file, owner, symbol),
			}
		}
		if len(candidates) > 1 {
			return map[string]interface{}{
				"error":      fmt.Sprintf("symbol %q is ambiguous in %s; nothing was cached (pass one of the qualified names)", symbol, rel),
				"file":       file,
				"symbol":     symbol,
				"candidates": candidates,
			}
		}
		// LoadSummary treats a summary as stale when this hash differs from the
		// symbol's current code hash, so it must be the code's, not the summary's.
		if code != "" {
			contentHash = search.ContentHash(code)
		}
	}

	err = db.IndexWrite(func(tx *sql.Tx) error {
		_, err := tx.Exec(
			`INSERT INTO summaries (file_path, symbol_name, summary_text, content_hash, project_path)
			 VALUES (?, ?, ?, ?, ?)
			 ON CONFLICT(file_path, symbol_name, project_path) DO UPDATE SET summary_text=excluded.summary_text, content_hash=excluded.content_hash, created_at=datetime('now')`,
			file, qualified, summary, contentHash, projectPath)
		return err
	})
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	out := map[string]interface{}{"status": "cached", "file": file, "symbol": name}
	if qualified != name {
		out["qualified_name"] = qualified
	}
	return out
}

// similarSymbols lists up to five symbols in file whose name contains the last
// segment of symbol, to point a caller at a near miss.
func similarSymbols(conn *sql.DB, file, projectPath, symbol string) []string {
	seg := strings.ToLower(symbol)
	if i := strings.LastIndex(seg, "."); i >= 0 {
		seg = seg[i+1:]
	}
	out := []string{}
	if seg == "" {
		return out
	}
	rows, err := conn.Query(
		`SELECT DISTINCT name, COALESCE(fqn,'') FROM symbols WHERE file = ? AND project_path = ? AND INSTR(LOWER(name), ?) > 0 ORDER BY start_line LIMIT 5`,
		file, projectPath, seg)
	if err != nil {
		return out
	}
	defer rows.Close()
	for rows.Next() {
		var name, fqn string
		if rows.Scan(&name, &fqn) == nil {
			out = append(out, db.QualifiedName(fqn, file, name))
		}
	}
	return out
}
