package context

import (
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// Relevance floor settings and placeholder defaults (tuned by the threshold sweep).
const (
	settingRelevanceMinRelative = "relevance_min_relative"
	settingRelevanceVectorMin   = "relevance_vector_min"
	settingRelevanceCoverageMin = "relevance_coverage_min"
	envRelevanceMinRelative     = "AST_RELEVANCE_MIN_RELATIVE"
	envRelevanceVectorMin       = "AST_RELEVANCE_VECTOR_MIN"
	envRelevanceCoverageMin     = "AST_RELEVANCE_COVERAGE_MIN"
	defaultMinRelative          = 0.35
	defaultVectorMin            = 0.45
	defaultCoverageMin          = 0.5
)

// FloorConfig holds the relevance floor thresholds.
type FloorConfig struct {
	// MinRelative withholds hits scoring below this fraction of the top hit's score.
	MinRelative float64
	// VectorMin is the cosine similarity at which a top hit counts as a match on its own.
	VectorMin float64
	// CoverageMin is the query term coverage at which a top hit counts as a match on its own.
	CoverageMin float64
}

// LoadFloorConfig reads the floor thresholds from env, then settings, then defaults.
func LoadFloorConfig() FloorConfig {
	return FloorConfig{
		MinRelative: db.SettingFloat(settingRelevanceMinRelative, envRelevanceMinRelative, defaultMinRelative),
		VectorMin:   db.SettingFloat(settingRelevanceVectorMin, envRelevanceVectorMin, defaultVectorMin),
		CoverageMin: db.SettingFloat(settingRelevanceCoverageMin, envRelevanceCoverageMin, defaultCoverageMin),
	}
}

// ApplyRelevanceFloor splits scored into hits at or above MinRelative of the top score and
// the withheld rest, preserving order. The top hit is always kept; the caller decides
// whether a weak top hit becomes a no-match. Without a positive top score nothing is withheld.
func ApplyRelevanceFloor(scored []search.ScoredResult, cfg FloorConfig) (kept, withheld []search.ScoredResult) {
	top := topIndex(scored)
	if top < 0 || scored[top].Score <= 0 {
		return scored, nil
	}
	topScore := scored[top].Score
	for i, r := range scored {
		if i == top || r.Score/topScore >= cfg.MinRelative {
			kept = append(kept, r)
		} else {
			withheld = append(withheld, r)
		}
	}
	return kept, withheld
}

// WeakMatch reports whether the top hit is too weak to count as a match, and the best
// evidence it has (the larger of its vector similarity and term coverage). The top hit is
// strong when its vector similarity reaches VectorMin or its query term coverage reaches
// CoverageMin. An empty list is weak with best 0.
//
// Fused hybrid results do not record which lists a hit came from: a hit's data map is the
// first list's (BM25 when in both), so similarity is only present for vector-only hits.
func WeakMatch(scored []search.ScoredResult, query string, cfg FloorConfig) (bool, float64) {
	top := topIndex(scored)
	if top < 0 {
		return true, 0
	}
	data := scored[top].Data
	sim, hasSim := similarity(data)
	cov := search.TermCoverage(query, coverageText(data))
	best := cov
	if hasSim && sim > best {
		best = sim
	}
	if hasSim && sim >= cfg.VectorMin {
		return false, best
	}
	return cov < cfg.CoverageMin, best
}

// topIndex is the index of the highest-scoring hit (first on ties), or -1 when empty.
func topIndex(scored []search.ScoredResult) int {
	top := -1
	for i, r := range scored {
		if top < 0 || r.Score > scored[top].Score {
			top = i
		}
	}
	return top
}

func similarity(data map[string]any) (float64, bool) {
	switch v := data["similarity"].(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	default:
		return 0, false
	}
}

// coverageText is the text a hit's term coverage is measured on: its names, file base
// name, and any body already attached.
func coverageText(data map[string]any) string {
	var parts []string
	for _, k := range []string{"name", "qualified_name", "fqn", "skeleton", "source", "summary"} {
		if s, _ := data[k].(string); s != "" {
			parts = append(parts, s)
		}
	}
	if f, _ := data["file"].(string); f != "" {
		parts = append(parts, filepath.Base(f))
	}
	return strings.Join(parts, " ")
}
