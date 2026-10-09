package context

import (
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/coma-toast/ast-context-cache/internal/tokens"
)

func numberedLines(n int) []string {
	lines := make([]string, n)
	for i := range lines {
		lines[i] = "line " + strconv.Itoa(i+1)
	}
	return lines
}

func TestConservativeBaselineTokensMergesPerFile(t *testing.T) {
	lines := numberedLines(200)
	cache := map[string][]string{"/a.go": lines, "/b.go": lines}
	count := func(start, end int) int { return tokens.Count(strings.Join(lines[start-1:end], "\n")) }
	// Two overlapping widened spans in a.go merge into [1,90]; b.go's span clamps at the end.
	got := ConservativeBaselineTokens([]LineSpan{
		{File: "/a.go", Start: 50, End: 70},
		{File: "/a.go", Start: 10, End: 15},
		{File: "/b.go", Start: 190, End: 195},
	}, cache)
	assert.Equal(t, count(1, 90)+count(170, 200), got)
	// Spans far apart in one file are counted separately, not as one block.
	got = ConservativeBaselineTokens([]LineSpan{{File: "/a.go", Start: 30, End: 30}, {File: "/a.go", Start: 150, End: 150}}, cache)
	assert.Equal(t, count(10, 50)+count(130, 170), got)
}

func TestConservativeBaselineTokensSkipsUnknownSpans(t *testing.T) {
	assert.Zero(t, ConservativeBaselineTokens([]LineSpan{{File: "", Start: 1, End: 2}, {File: "/x", Start: 0, End: 0}}, map[string][]string{}))
	assert.Zero(t, ConservativeBaselineTokens([]LineSpan{{File: "/does/not/exist.go", Start: 1, End: 3}}, map[string][]string{}))
}

func TestApplyToOmitsConservativeBaseline(t *testing.T) {
	resp := map[string]interface{}{}
	SavingsMeta{TokensUsed: 10, ConservativeBaseline: 999}.ApplyTo(resp)
	for k, v := range resp {
		assert.NotEqual(t, 999, v, k)
	}
}
