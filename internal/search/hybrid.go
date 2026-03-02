package search

import (
	"sort"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

const rrfK = 60 // reciprocal rank fusion constant

// HybridSearch runs BM25 and vector search in parallel, merges via RRF.
// If emb is nil, falls back to BM25 only.
func HybridSearch(query, projectPath string, emb *embedder.Embedder, limit int) []ScoredResult {
	bm25Results := BM25Search(query, projectPath)

	var vectorResults []ScoredResult
	if emb != nil && Cache.Count(projectPath) > 0 {
		queryVec, err := emb.EmbedSingle(query)
		if err == nil {
			vectorResults = Cache.Search(queryVec, projectPath, "", limit*2)
		}
	}

	if len(vectorResults) == 0 {
		if len(bm25Results) > limit {
			bm25Results = bm25Results[:limit]
		}
		return bm25Results
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

	merged := make([]ScoredResult, 0, len(seen))
	for _, e := range seen {
		merged = append(merged, ScoredResult{Data: e.data, Score: e.score})
	}
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	if len(merged) > limit {
		merged = merged[:limit]
	}
	return merged
}

func resultKey(r ScoredResult) string {
	name, _ := r.Data["name"].(string)
	file, _ := r.Data["file"].(string)
	kind, _ := r.Data["kind"].(string)
	return file + "|" + name + "|" + kind
}
