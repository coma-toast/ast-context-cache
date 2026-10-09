package context

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestModeRank(t *testing.T) {
	t.Parallel()
	tests := []struct {
		mode string
		want int
	}{
		{"locations", 0}, {"summary", 1}, {"skeleton", 2}, {"full", 3}, {"edit", 3}, {"auto", 2}, {"", 2},
	}
	for _, tt := range tests {
		t.Run(tt.mode, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, ModeRank(tt.mode))
		})
	}
}

func TestEffectiveModeV2(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, mode string
		rank       int
		want       string
	}{
		{name: "auto top", mode: "auto", rank: 0, want: "full"},
		{name: "auto third", mode: "auto", rank: 2, want: "full"},
		{name: "auto fourth", mode: "auto", rank: 3, want: "skeleton"},
		{name: "auto tail", mode: "auto", rank: 40, want: "skeleton"},
		{name: "explicit summary", mode: "summary", rank: 0, want: "summary"},
		{name: "explicit skeleton", mode: "skeleton", rank: 0, want: "skeleton"},
		{name: "edit", mode: "edit", rank: 9, want: "edit"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, EffectiveModeV2(tt.mode, tt.rank))
		})
	}
}
