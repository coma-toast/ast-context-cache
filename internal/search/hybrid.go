package search

import (
	"sort"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

const rrfK = 60 // reciprocal rank fusion constant

// HybridSearchMetrics counts candidates at each hybrid-search stage (after filters, before final limit).
type HybridSearchMetrics struct {
	BM25Candidates   int `json:"bm25_candidates"`
	VectorCandidates int `json:"vector_candidates"`
	HybridAfterFuse  int `json:"hybrid_after_fuse"`
}

// HybridSearch runs BM25 and vector search in parallel, merges via RRF.
// If emb is nil, falls back to BM25 only. filters may be nil.
func HybridSearch(query, projectPath string, emb embedder.Interface, limit int, filters *SearchFilters) ([]ScoredResult, *HybridSearchMetrics) {
	metrics := &HybridSearchMetrics{}
	bm25Results := BM25Search(query, projectPath, filters)
	metrics.BM25Candidates = len(bm25Results)

	var vectorResults []ScoredResult
	if emb != nil && Cache.Count(projectPath) > 0 {
		queryVec, err := emb.EmbedSingle(query)
		if err == nil {
			vectorResults = Cache.Search(queryVec, projectPath, "", limit*2, filters)
		}
	}
	metrics.VectorCandidates = len(vectorResults)

	if len(vectorResults) == 0 {
		if len(bm25Results) > limit {
			bm25Results = bm25Results[:limit]
		}
		metrics.HybridAfterFuse = len(bm25Results)
		return bm25Results, metrics
	}

	type fusedEntry struct {
		key   string
		data  map[string]interface{}
		score float64
	}

	seen := map[string]*fusedEntry{}

	for rank, r := range bm25Results {
		key := resultKey(r)
		if e, ok := seen[key]; ok {
			e.score += 1.0 / float64(rrfK+rank+1)
		} else {
			seen[key] = &fusedEntry{
				key:   key,
				data:  r.Data,
				score: 1.0 / float64(rrfK+rank+1),
			}
		}
	}

	for rank, r := range vectorResults {
		key := resultKey(r)
		if e, ok := seen[key]; ok {
			e.score += 1.0 / float64(rrfK+rank+1)
		} else {
			seen[key] = &fusedEntry{
				key:   key,
				data:  r.Data,
				score: 1.0 / float64(rrfK+rank+1),
			}
		}
	}

	// Fuse in sorted-key order and break score ties on the key, so tied hits
	// come back in the same order on every run.
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fused := make([]*fusedEntry, len(keys))
	for i, k := range keys {
		fused[i] = seen[k]
	}
	sort.SliceStable(fused, func(i, j int) bool {
		if fused[i].score != fused[j].score {
			return fused[i].score > fused[j].score
		}
		return fused[i].key < fused[j].key
	})
	merged := make([]ScoredResult, len(fused))
	for i, e := range fused {
		merged[i] = ScoredResult{Data: e.data, Score: e.score}
	}

	metrics.HybridAfterFuse = len(merged)

	if len(merged) > limit {
		merged = merged[:limit]
	}
	return merged, metrics
}

// resultKey identifies a hit's symbol for fusion. Same-named methods of
// different classes in one file differ only by qualified_name.
func resultKey(r ScoredResult) string {
	name, _ := r.Data["name"].(string)
	if q, _ := r.Data["qualified_name"].(string); q != "" {
		name = q
	}
	file, _ := r.Data["file"].(string)
	kind, _ := r.Data["kind"].(string)
	return file + "|" + name + "|" + kind
}
