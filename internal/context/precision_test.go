package context

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func scoredNames(rs []search.ScoredResult) []string {
	var out []string
	for _, r := range rs {
		out = append(out, r.Data["name"].(string))
	}
	return out
}

func hit(name string, score float64, extra ...any) search.ScoredResult {
	data := map[string]any{"name": name, "kind": "function", "file": "/p/" + name + ".go"}
	for i := 0; i+1 < len(extra); i += 2 {
		data[extra[i].(string)] = extra[i+1]
	}
	return search.ScoredResult{Data: data, Score: score}
}

func TestApplyRelevanceFloor(t *testing.T) {
	t.Parallel()
	cfg := FloorConfig{MinRelative: 0.5}
	tests := []struct {
		name         string
		scored       []search.ScoredResult
		cfg          FloorConfig
		kept, withld []string
	}{
		{name: "empty", cfg: cfg},
		{name: "single", scored: []search.ScoredResult{hit("a", 1)}, cfg: cfg, kept: []string{"a"}},
		{name: "relative cut keeps order", scored: []search.ScoredResult{hit("a", 10), hit("b", 4), hit("c", 6), hit("d", 5)}, cfg: cfg, kept: []string{"a", "c", "d"}, withld: []string{"b"}},
		{name: "unsorted top", scored: []search.ScoredResult{hit("a", 2), hit("b", 10)}, cfg: cfg, kept: []string{"b"}, withld: []string{"a"}},
		{name: "top kept over 1.0 floor", scored: []search.ScoredResult{hit("a", 3), hit("b", 3)}, cfg: FloorConfig{MinRelative: 2}, kept: []string{"a"}, withld: []string{"b"}},
		{name: "zero top keeps all", scored: []search.ScoredResult{hit("a", 0), hit("b", 0)}, cfg: cfg, kept: []string{"a", "b"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			kept, withheld := ApplyRelevanceFloor(tt.scored, tt.cfg)
			assert.Equal(t, tt.kept, scoredNames(kept))
			assert.Equal(t, tt.withld, scoredNames(withheld))
		})
	}
}

func TestWeakMatch(t *testing.T) {
	t.Parallel()
	cfg := FloorConfig{VectorMin: 0.45, CoverageMin: 0.5}
	tests := []struct {
		name     string
		scored   []search.ScoredResult
		query    string
		wantWeak bool
		wantBest float64
	}{
		{name: "empty", query: "x", wantWeak: true},
		{name: "vector similarity strong", scored: []search.ScoredResult{hit("Unrelated", 1, "similarity", 0.6)}, query: "load model", wantBest: 0.6},
		{name: "coverage strong", scored: []search.ScoredResult{hit("LoadModel", 1)}, query: "loadmodel config", wantBest: 0.5},
		{name: "coverage from skeleton", scored: []search.ScoredResult{hit("x", 1, "skeleton", "func x(cfg Config) error")}, query: "config error", wantBest: 1},
		{name: "weak similarity and coverage", scored: []search.ScoredResult{hit("Widget", 1, "similarity", 0.3)}, query: "quantum flux", wantWeak: true, wantBest: 0.3},
		{name: "uses top scored hit", scored: []search.ScoredResult{hit("Widget", 1), hit("Quantum", 5)}, query: "quantum", wantBest: 1},
		{name: "file base name counts", scored: []search.ScoredResult{hit("x", 1, "file", "/p/other.go")}, query: "other", wantBest: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			weak, best := WeakMatch(tt.scored, tt.query, cfg)
			assert.Equal(t, tt.wantWeak, weak)
			assert.InDelta(t, tt.wantBest, best, 1e-9)
		})
	}
}

func TestLoadFloorConfig(t *testing.T) {
	dbtest.Init(t)
	defaults := FloorConfig{MinRelative: defaultMinRelative, VectorMin: defaultVectorMin, CoverageMin: defaultCoverageMin}
	tests := []struct {
		name     string
		env      map[string]string
		settings map[string]string
		want     FloorConfig
	}{
		{name: "defaults", want: defaults},
		{name: "settings", settings: map[string]string{settingRelevanceMinRelative: "0.2", settingRelevanceVectorMin: "0.7", settingRelevanceCoverageMin: "0.9"}, want: FloorConfig{MinRelative: 0.2, VectorMin: 0.7, CoverageMin: 0.9}},
		{name: "env wins", env: map[string]string{envRelevanceMinRelative: "0.1"}, settings: map[string]string{settingRelevanceMinRelative: "0.2"}, want: FloorConfig{MinRelative: 0.1, VectorMin: defaultVectorMin, CoverageMin: defaultCoverageMin}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, k := range []string{envRelevanceMinRelative, envRelevanceVectorMin, envRelevanceCoverageMin} {
				t.Setenv(k, tt.env[k])
			}
			for _, k := range []string{settingRelevanceMinRelative, settingRelevanceVectorMin, settingRelevanceCoverageMin} {
				require.NoError(t, db.SetSetting(k, tt.settings[k]))
			}
			assert.Equal(t, tt.want, LoadFloorConfig())
		})
	}
}
