package search

import (
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// Search queries are assembled at runtime from these fragments: the project scope
// and optional filters are spliced in between the select and the order/limit tail.
const (
	selectFTSSymbolsQuery = `
			SELECT s.name, s.kind, s.file, s.start_line, s.end_line, COALESCE(s.fqn,''), f.rank
			FROM symbols_fts f
			JOIN symbols s ON f.rowid = s.id
			WHERE `
	ftsMatchClause   = ` AND symbols_fts MATCH ?`
	ftsOrderByClause = `
			ORDER BY f.rank, s.id
			LIMIT 100`
	selectTrigramSymbolsQuery = `
		SELECT s.name, s.kind, s.file, s.start_line, s.end_line, COALESCE(s.fqn,''), t.rank
		FROM symbols_trigram t
		JOIN symbols s ON t.rowid = s.id
		WHERE `
	trigramMatchClause   = ` AND symbols_trigram MATCH ?`
	trigramOrderByClause = `
		ORDER BY t.rank, s.id
		LIMIT 100`
	selectFallbackSymbolsQuery = "SELECT s.name, s.kind, s.file, s.start_line, s.end_line, COALESCE(s.fqn,'') FROM symbols s WHERE "
	fallbackTermClause         = "(LOWER(s.name) LIKE ? OR LOWER(s.fqn) LIKE ? OR LOWER(s.code) LIKE ?)"
	fallbackLimitClause        = " ORDER BY s.id LIMIT 100"
	sqlAnd                     = " AND "
	sqlOr                      = " OR "
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
		q := selectFTSSymbolsQuery
		scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
		q += scopeFrag + ftsMatchClause
		args := append(scopeArgs, ftsQuery)
		if frag, extra := symbolFilterSQL(filters, projectPath); frag != "" {
			q += sqlAnd + frag
			args = append(args, extra...)
		}
		q += ftsOrderByClause
		rows, err := conn.Query(q, args...)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var name, kind, file, fqn string
				var startLine, endLine int
				var rank float64
				rows.Scan(&name, &kind, &file, &startLine, &endLine, &fqn, &rank)
				scored = append(scored, ScoredResult{
					Data:  symbolResult(name, kind, file, fqn, startLine, endLine),
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

// symbolResult is one symbol search hit. A member (a method, or a class nested
// in another) also carries its qualified_name (LlamaCppClient.load_model), since
// its bare name alone doesn't say which class it belongs to.
func symbolResult(name, kind, file, fqn string, startLine, endLine int) map[string]interface{} {
	data := map[string]interface{}{
		"name": name, "kind": kind, "file": file,
		"start_line": startLine, "end_line": endLine,
	}
	if q := db.QualifiedName(fqn, file, name); q != name {
		data["qualified_name"] = q
	}
	return data
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
	q := selectTrigramSymbolsQuery
	scopeFrag, scopeArgs := projectlinks.ScopeSQL("s", projectPath)
	q += scopeFrag + trigramMatchClause
	args := append(scopeArgs, tqQuery)
	if frag, extra := symbolFilterSQL(filters, projectPath); frag != "" {
		q += sqlAnd + frag
		args = append(args, extra...)
	}
	q += trigramOrderByClause
	rows, err := conn.Query(q, args...)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var scored []ScoredResult
	for rows.Next() {
		var name, kind, file, fqn string
		var startLine, endLine int
		var rank float64
		rows.Scan(&name, &kind, &file, &startLine, &endLine, &fqn, &rank)
		scored = append(scored, ScoredResult{
			Data:  symbolResult(name, kind, file, fqn, startLine, endLine),
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
		conditions = append(conditions, fallbackTermClause)
		sqlArgs = append(sqlArgs, pattern, pattern, pattern)
	}
	where := scopeFrag
	if len(conditions) > 0 {
		where += sqlAnd + "(" + strings.Join(conditions, sqlOr) + ")"
	}
	if frag, extra := symbolFilterSQL(filters, projectPath); frag != "" {
		where += sqlAnd + frag
		sqlArgs = append(sqlArgs, extra...)
	}
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	rows, err := conn.Query(selectFallbackSymbolsQuery+where+fallbackLimitClause, sqlArgs...)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var scored []ScoredResult
	for rows.Next() {
		var name, kind, file, fqn string
		var startLine, endLine int
		rows.Scan(&name, &kind, &file, &startLine, &endLine, &fqn)
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
			Data:  symbolResult(name, kind, file, fqn, startLine, endLine),
			Score: s,
		})
	}
	sort.SliceStable(scored, func(i, j int) bool { return LessScored(scored[i], scored[j]) })
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
