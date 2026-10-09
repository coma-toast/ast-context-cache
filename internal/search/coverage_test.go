package search

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTermCoverage(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, query, text string
		want              float64
	}{
		{name: "empty query", query: "", text: "anything", want: 0},
		{name: "all terms", query: "vector cache", text: "func VectorCache() cache", want: 1},
		{name: "prefix match", query: "embed", text: "EmbedSingle embedder", want: 1},
		{name: "half", query: "parse config", text: "func parseArgs()", want: 0.5},
		{name: "stopwords dropped", query: "how to parse the config", text: "parse", want: 0.5},
		{name: "only stopwords kept", query: "the", text: "the end", want: 1},
		{name: "multi-piece term needs every piece", query: "foo.bar", text: "foo baz", want: 0},
		{name: "duplicate terms counted once", query: "load load model", text: "load", want: 0.5},
		{name: "no match", query: "quantum", text: "func Widget()", want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.InDelta(t, tt.want, TermCoverage(tt.query, tt.text), 1e-9)
		})
	}
}
