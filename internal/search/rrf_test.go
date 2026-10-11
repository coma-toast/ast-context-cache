package search

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRRF(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		lists [][]string
		want  map[string]float64
	}{
		{name: "no lists", want: map[string]float64{}},
		{name: "single list", lists: [][]string{{"a", "b"}}, want: map[string]float64{"a": 1.0 / 61, "b": 1.0 / 62}},
		{name: "ids in both lists add up", lists: [][]string{{"a", "b"}, {"b", "c"}}, want: map[string]float64{"a": 1.0 / 61, "b": 1.0/62 + 1.0/61, "c": 1.0 / 62}},
		{name: "empty and repeated ids skipped", lists: [][]string{{"", "a", "a"}}, want: map[string]float64{"a": 1.0 / 62}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := RRF(tc.lists...)
			assert.Len(t, got, len(tc.want))
			for id, score := range tc.want {
				assert.InDelta(t, score, got[id], 1e-12, id)
			}
		})
	}
}
