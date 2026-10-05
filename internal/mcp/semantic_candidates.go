package mcp

import (
	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// semanticCandidates ranks vectors for query through the shared candidate cache. A
// cache hit skips embedding the query as well as the scan; a failed embed is not cached.
func semanticCandidates(query, projectPath, docType string, limit int, filters *search.SearchFilters) (scored []search.ScoredResult, cacheHit bool, err error) {
	q := context.CandidateQuery{Stage: "semantic:vector", Query: query, ProjectPath: projectPath, DocType: docType, Limit: limit, Filters: filters}
	scored, _, cacheHit, err = context.RankedCandidates(q, func() ([]search.ScoredResult, *search.HybridSearchMetrics, error) {
		queryVec, err := emb.EmbedSingle(query)
		RecordEmbed()
		if err != nil {
			return nil, nil, err
		}
		return search.Cache.Search(queryVec, projectPath, docType, limit, filters), nil, nil
	})
	return scored, cacheHit, err
}
