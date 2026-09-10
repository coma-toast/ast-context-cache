package search

import (
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

type ScoredResult struct {
	Data  map[string]interface{}
	Score float64
}

func BM25Search(query, projectPath string, filters *SearchFilters) []ScoredResult {
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	terms := strings.Fields(strings.ToLower(query))
	var scored []ScoredResult

	ftsQuery := BuildFTSQuery(terms)
	if ftsQuery != "" {
		q := `
			SELECT s.name, s.kind, s.file, s.start_line, s.end_line, f.rank
			FROM symbols_fts f
			JOIN symbols s ON f.rowid = s.id
			WHERE `
		scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
		q += scopeFrag + ` AND symbols_fts MATCH ?`
		args := append(scopeArgs, ftsQuery)
		if frag, extra := symbolFilterSQL(filters, projectPath); frag != "" {
			q += " AND " + frag
			args = append(args, extra...)
		}
		q += `
			ORDER BY f.rank
			LIMIT 100`
		rows, err := conn.Query(q, args...)
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
		scored = TrigramSearch(terms, projectPath, filters)
	}

	if len(scored) == 0 {
		scored = FallbackSearch(terms, projectPath, filters)
	}

	scored = filterScoredResults(scored, projectPath, filters)
	return scored
}

// TrigramSearch matches terms as substrings anywhere in a symbol's name or fqn, using
// the trigram-tokenized symbols_trigram index (see schema_index.go). This is the fast,
// indexed path for queries that BuildFTSQuery's prefix-only matching misses — e.g.
// "Cache" against "VectorCache", which unicode61 tokenizes as a single token and so
// never matches a "Cache*" prefix query. Falls back to FallbackSearch's LIKE scan only
// when this also finds nothing (or every term is under 3 characters, the trigram
// tokenizer's minimum matchable substring length).
func TrigramSearch(terms []string, projectPath string, filters *SearchFilters) []ScoredResult {
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	tqQuery := BuildTrigramQuery(terms)
	if tqQuery == "" {
		return nil
	}
	q := `
		SELECT s.name, s.kind, s.file, s.start_line, s.end_line, t.rank
		FROM symbols_trigram t
		JOIN symbols s ON t.rowid = s.id
		WHERE `
	scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
	q += scopeFrag + ` AND symbols_trigram MATCH ?`
	args := append(scopeArgs, tqQuery)
	if frag, extra := symbolFilterSQL(filters, projectPath); frag != "" {
		q += " AND " + frag
		args = append(args, extra...)
	}
	q += `
		ORDER BY t.rank
		LIMIT 100`
	rows, err := conn.Query(q, args...)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var scored []ScoredResult
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
	return scored
}

// BuildTrigramQuery quotes each term as a literal substring phrase and ORs them
// together, mirroring BuildFTSQuery's "any of these terms" semantics. Terms under 3
// characters are dropped since the trigram tokenizer can never match them.
func BuildTrigramQuery(terms []string) string {
	var parts []string
	for _, t := range terms {
		if len(t) < 3 {
			continue
		}
		parts = append(parts, `"`+strings.ReplaceAll(t, `"`, `""`)+`"`)
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " OR ")
}

func FallbackSearch(terms []string, projectPath string, filters *SearchFilters) []ScoredResult {
	var conditions []string
	scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
	var sqlArgs []interface{}
	sqlArgs = append(sqlArgs, scopeArgs...)
	for _, term := range terms {
		pattern := "%" + term + "%"
		conditions = append(conditions, "(LOWER(s.name) LIKE ? OR LOWER(s.fqn) LIKE ? OR LOWER(s.code) LIKE ?)")
		sqlArgs = append(sqlArgs, pattern, pattern, pattern)
	}
	where := scopeFrag
	if len(conditions) > 0 {
		where += " AND (" + strings.Join(conditions, " OR ") + ")"
	}
	if frag, extra := symbolFilterSQL(filters, projectPath); frag != "" {
		where += " AND " + frag
		sqlArgs = append(sqlArgs, extra...)
	}
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	rows, err := conn.Query("SELECT s.name, s.kind, s.file, s.start_line, s.end_line FROM symbols s WHERE "+where+" LIMIT 100", sqlArgs...)
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

// QueryTerms splits a user query into lowercase terms for FTS query building.
func QueryTerms(query string) []string {
	return strings.Fields(strings.ToLower(query))
}
