package docs

import (
	"os"
	"strconv"
	"strings"
	"unicode"
)

// Relevance floor for doc search.
//
// The score search_docs returns in hybrid mode is Reciprocal Rank Fusion:
// sum over lists of 1/(docRRFK+rank+1). It only encodes rank, not relevance: the top hit
// of one list always scores 1/61 ≈ 0.0164 and the top hit of both 2/61 ≈ 0.0328, whether
// or not it matches. That is exactly the 0.016–0.03 band unrelated hits came back with,
// so a floor on the fused score cannot tell a match from noise. The floor is applied to
// the two absolute signals before fusion instead:
//
//   - Lexical: the FTS query is term1* OR term2* OR ..., so one common word ("module",
//     "attribute") is enough to match. A hit must contain at least half of the query's
//     content terms (prefix match on unicode61-style tokens, like the FTS query).
//   - Semantic: vector hits are cosine similarity. A hit with no lexical support must
//     reach docMinVectorSimilarity. The default targets nomic-embed-text (768-d, used
//     without task prefixes here), where unrelated text still scores ~0.3–0.5; override
//     with AST_DOCS_MIN_VECTOR_SIMILARITY for another embedder.
const (
	docMinTermCoverage            = 0.5
	defaultDocMinVectorSimilarity = 0.6
)

// docStopwords are ignored when measuring term coverage; with prefix matching, short
// function words match nearly every section and would inflate coverage.
var docStopwords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true, "be": true,
	"by": true, "can": true, "do": true, "does": true, "for": true, "from": true, "how": true,
	"i": true, "in": true, "is": true, "it": true, "of": true, "on": true, "or": true,
	"the": true, "to": true, "what": true, "when": true, "with": true,
}

func docMinVectorSimilarity() float64 {
	if v := os.Getenv("AST_DOCS_MIN_VECTOR_SIMILARITY"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= -1 && f <= 1 {
			return f
		}
	}
	return defaultDocMinVectorSimilarity
}

func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// coverageTerms turns a query into content terms, each a list of token pieces
// ("__getattr__" -> [getattr], "foo.bar" -> [foo bar]). Stopwords are dropped unless
// the query has nothing else.
func coverageTerms(query string) [][]string {
	var all, content [][]string
	seen := map[string]bool{}
	for _, raw := range strings.Fields(query) {
		pieces := tokenize(raw)
		if len(pieces) == 0 {
			continue
		}
		key := strings.Join(pieces, " ")
		if seen[key] {
			continue
		}
		seen[key] = true
		all = append(all, pieces)
		if len(pieces) > 1 || !docStopwords[pieces[0]] {
			content = append(content, pieces)
		}
	}
	if len(content) == 0 {
		return all
	}
	return content
}

// termCoverage is the fraction of query terms present in the section's title+content;
// a term is present when every piece prefixes some token (mirrors the FTS term* query).
func termCoverage(terms [][]string, e DocEntry) float64 {
	if len(terms) == 0 {
		return 0
	}
	tokens := tokenize(e.Title + " " + e.Content)
	hasPrefix := func(p string) bool {
		for _, t := range tokens {
			if strings.HasPrefix(t, p) {
				return true
			}
		}
		return false
	}
	matched := 0
	for _, pieces := range terms {
		ok := true
		for _, p := range pieces {
			if !hasPrefix(p) {
				ok = false
				break
			}
		}
		if ok {
			matched++
		}
	}
	return float64(matched) / float64(len(terms))
}

// applyLexicalFloor annotates FTS/LIKE hits with coverage and drops those below the floor.
func applyLexicalFloor(terms [][]string, hits []ScoredDoc) []ScoredDoc {
	var kept []ScoredDoc
	for _, h := range hits {
		h.TermCoverage = termCoverage(terms, h.Entry)
		if h.TermCoverage >= docMinTermCoverage {
			kept = append(kept, h)
		}
	}
	return kept
}

// applyVectorFloor keeps vector hits that are either similar enough on their own or
// lexically supported (FTS may have cut them off at its limit).
func applyVectorFloor(terms [][]string, hits []ScoredDoc) []ScoredDoc {
	minSim := docMinVectorSimilarity()
	var kept []ScoredDoc
	for _, h := range hits {
		h.Similarity = h.Score
		h.TermCoverage = termCoverage(terms, h.Entry)
		if h.Similarity >= minSim || h.TermCoverage >= docMinTermCoverage {
			kept = append(kept, h)
		}
	}
	return kept
}

// countBelowFloor is how many distinct candidate sections the floor removed.
func countBelowFloor(candidates, kept [][]ScoredDoc) int {
	ids := func(lists [][]ScoredDoc) map[int]bool {
		m := map[int]bool{}
		for _, l := range lists {
			for _, s := range l {
				m[s.Entry.ID] = true
			}
		}
		return m
	}
	keptIDs := ids(kept)
	n := 0
	for id := range ids(candidates) {
		if !keptIDs[id] {
			n++
		}
	}
	return n
}
