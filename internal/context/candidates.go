package context

import (
	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// CandidateQuery identifies one ranked search for the shared candidate cache.
type CandidateQuery struct {
	Stage       string // tool and backend, e.g. "capsule:hybrid"
	Query       string
	ProjectPath string
	DocType     string
	Limit       int
	Filters     *search.SearchFilters
}

// CandidateSearch runs one ranked search; metrics may be nil.
type CandidateSearch func() ([]search.ScoredResult, *search.HybridSearchMetrics, error)

// RankedCandidates returns q's ranked candidates from the shared cache, or runs
// compute and caches them, when feature_shared_query_cache is on — with or without
// a session, since dedup, mode and budget are applied per call afterwards. The
// results are the caller's own copy. hit reports whether the cache answered.
func RankedCandidates(q CandidateQuery, compute CandidateSearch) (results []search.ScoredResult, metrics *search.HybridSearchMetrics, hit bool, err error) {
	if !flags.Enabled(flags.KeySharedQueryCache) {
		results, metrics, err = compute()
		return results, nonNilMetrics(metrics), false, err
	}
	filtersKey := ""
	if q.Filters != nil {
		filtersKey = q.Filters.CacheKey()
	}
	key := cache.Key(q.Stage, q.Query, q.ProjectPath, filtersKey, q.DocType, q.Limit)
	results, m, hit, err := cache.Candidates.Fetch(key, q.ProjectPath, func() ([]search.ScoredResult, cache.PipelineMetrics, error) {
		r, m, err := compute()
		return r, *nonNilMetrics(m), err
	})
	return results, &m, hit, err
}

// RankedHybrid is RankedCandidates over search.HybridSearch (q.Limit candidates).
func RankedHybrid(q CandidateQuery, emb embedder.Interface) ([]search.ScoredResult, *search.HybridSearchMetrics, bool) {
	results, metrics, hit, _ := RankedCandidates(q, func() ([]search.ScoredResult, *search.HybridSearchMetrics, error) {
		r, m := search.HybridSearch(q.Query, q.ProjectPath, emb, q.Limit, q.Filters)
		return r, m, nil
	})
	return results, metrics, hit
}

func nonNilMetrics(m *search.HybridSearchMetrics) *search.HybridSearchMetrics {
	if m == nil {
		return &search.HybridSearchMetrics{}
	}
	return m
}
