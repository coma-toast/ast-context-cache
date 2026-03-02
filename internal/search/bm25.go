package search

import (
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

type ScoredResult struct {
	Data  map[string]interface{}
	Score float64
}

func BM25Search(query, projectPath string) []ScoredResult {
	terms := strings.Fields(strings.ToLower(query))
	var scored []ScoredResult

	ftsQuery := BuildFTSQuery(terms)
	if ftsQuery != "" {
		rows, err := db.DB.Query(`
			SELECT s.name, s.kind, s.file, s.start_line, s.end_line, f.rank
			FROM symbols_fts f
			JOIN symbols s ON f.rowid = s.id
			WHERE s.project_path = ? AND symbols_fts MATCH ?
			ORDER BY f.rank
			LIMIT 100`, projectPath, ftsQuery)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var name, kind, file string
				var startLine, endLine int
				var rank float64
				rows.Scan(&name, &kind, &file, &startLine, &endLine, &rank)
				scored = append(scored, ScoredResult{
					Data: map[string]interface{}{
						"name": name, "kind": kind, "file": file,
						"start_line": startLine, "end_line": endLine,
					},
					Score: -rank,
				})
			}
		}
	}

	if len(scored) == 0 {
		scored = FallbackSearch(terms, projectPath)
	}

	return scored
}

func FallbackSearch(terms []string, projectPath string) []ScoredResult {
	var conditions []string
	var sqlArgs []interface{}
	sqlArgs = append(sqlArgs, projectPath)
	for _, term := range terms {
		pattern := "%" + term + "%"
		conditions = append(conditions, "(LOWER(name) LIKE ? OR LOWER(fqn) LIKE ? OR LOWER(code) LIKE ?)")
		sqlArgs = append(sqlArgs, pattern, pattern, pattern)
	}
	where := "project_path = ?"
	if len(conditions) > 0 {
		where += " AND (" + strings.Join(conditions, " OR ") + ")"
	}
	rows, err := db.DB.Query("SELECT name, kind, file, start_line, end_line FROM symbols WHERE "+where+" LIMIT 100", sqlArgs...)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var scored []ScoredResult
	for rows.Next() {
		var name, kind, file string
		var startLine, endLine int
		rows.Scan(&name, &kind, &file, &startLine, &endLine)
		s := 0.0
		nameLower := strings.ToLower(name)
		for _, t := range terms {
			if nameLower == t {
				s += 10
			} else if strings.HasPrefix(nameLower, t) {
				s += 5
			} else if strings.Contains(nameLower, t) {
				s += 3
			} else {
				s += 1
			}
		}
		scored = append(scored, ScoredResult{
			Data: map[string]interface{}{
				"name": name, "kind": kind, "file": file,
				"start_line": startLine, "end_line": endLine,
			},
			Score: s,
		})
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})
	return scored
}

func BuildFTSQuery(terms []string) string {
	if len(terms) == 0 {
		return ""
	}
	var parts []string
	for _, t := range terms {
		cleaned := strings.Map(func(r rune) rune {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
				return r
			}
			return -1
		}, t)
		if cleaned != "" {
			parts = append(parts, cleaned+"*")
		}
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " OR ")
}
