package search

import (
	"strings"
	"unicode"
)

// coverageStopwords are ignored when measuring term coverage; with prefix matching, short
// function words match nearly every text and would inflate coverage. Mirrors docs/relevance.go.
var coverageStopwords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true, "be": true,
	"by": true, "can": true, "do": true, "does": true, "for": true, "from": true, "how": true,
	"i": true, "in": true, "is": true, "it": true, "of": true, "on": true, "or": true,
	"the": true, "to": true, "what": true, "when": true, "with": true,
}

// TermCoverage is the fraction of the query's content terms present in text. A term
// ("__getattr__" -> [getattr], "foo.bar" -> [foo bar]) is present when every piece
// prefixes some token of text, mirroring an FTS term* query. Stopwords are dropped
// unless the query has nothing else. An empty query covers nothing.
func TermCoverage(query, text string) float64 {
	terms := coverageTerms(query)
	if len(terms) == 0 {
		return 0
	}
	toks := coverageTokenize(text)
	hasPrefix := func(p string) bool {
		for _, t := range toks {
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

func coverageTokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func coverageTerms(query string) [][]string {
	var all, content [][]string
	seen := map[string]bool{}
	for _, raw := range strings.Fields(query) {
		pieces := coverageTokenize(raw)
		if len(pieces) == 0 {
			continue
		}
		key := strings.Join(pieces, " ")
		if seen[key] {
			continue
		}
		seen[key] = true
		all = append(all, pieces)
		if len(pieces) > 1 || !coverageStopwords[pieces[0]] {
			content = append(content, pieces)
		}
	}
	if len(content) == 0 {
		return all
	}
	return content
}
