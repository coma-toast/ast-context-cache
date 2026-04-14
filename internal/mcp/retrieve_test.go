package mcp

import (
	"encoding/json"
	"testing"
)

// Golden shape for retrieve stats JSON (regression guard for observability fields).
func TestRetrieveStatsJSONGolden(t *testing.T) {
	stats := RetrieveStats{
		CodeResults:        1,
		DocResults:         2,
		TotalTokens:        3,
		SearchTimeMs:       4.5,
		BM25Candidates:     6,
		VectorCandidates:   7,
		HybridAfterFuse:    8,
		AfterDedup:         9,
		ChunksInBudget:     10,
		TokensEstAllChunks: 11,
		CodeRetrieveMs:     12,
		DocsRetrieveMs:     13,
		DedupBudgetMs:      14,
	}
	b, err := json.Marshal(map[string]interface{}{"stats": stats})
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"stats":{"code_results":1,"doc_results":2,"total_tokens":3,"search_time_ms":4.5,"bm25_candidates":6,"vector_candidates":7,"hybrid_after_fuse":8,"after_dedup":9,"chunks_in_budget":10,"tokens_est_all_chunks":11,"code_retrieve_ms":12,"docs_retrieve_ms":13,"dedup_budget_ms":14}}`
	if string(b) != want {
		t.Fatalf("stats JSON mismatch:\ngot:  %s\nwant: %s", b, want)
	}
}
