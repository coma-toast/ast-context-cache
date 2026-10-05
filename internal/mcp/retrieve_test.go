package mcp

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
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

func retrieveFor(t *testing.T, project, sessionID string, budget int) RetrieveResult {
	t.Helper()
	args := map[string]interface{}{"query": "load_model", "include_docs": false, "token_budget": float64(budget)}
	if sessionID != "" {
		args["session_id"] = sessionID
	}
	out := HandleRetrieve(args, project)
	raw, ok := out["result"].(json.RawMessage)
	require.True(t, ok, "%v", out)
	var r RetrieveResult
	require.NoError(t, json.Unmarshal(raw, &r))
	return r
}

func chunkKeys(r RetrieveResult) []string {
	var out []string
	for _, c := range r.Chunks {
		out = append(out, c.QualifiedName)
	}
	return out
}

// Only chunks that fit the token budget are delivered, so only they are deduped
// on the session's next call; a trimmed one must still be returned then.
func TestRetrieveDedupsOnlyDeliveredChunks(t *testing.T) {
	project, _ := indexedPython(t, "clients.py", twoClientsPy)
	all := retrieveFor(t, project, "", 4000)
	require.Len(t, all.Chunks, 2)
	sid := t.Name()
	first := retrieveFor(t, project, sid, db.EstimateTokens(all.Chunks[0].Content))
	require.Len(t, first.Chunks, 1, "the budget fits one chunk")
	assert.Positive(t, first.Chunks[0].StartLine)
	delivered := first.Chunks[0].QualifiedName

	next := retrieveFor(t, project, sid, 4000)
	assert.NotContains(t, chunkKeys(next), delivered, "the delivered chunk is deduped")
	assert.Len(t, next.Chunks, 1, "the budget-trimmed chunk is not")
	assert.Equal(t, 1, next.Stats.DedupedCount)
	raw, _ := json.Marshal(next.Stats)
	assert.Contains(t, string(raw), `"deduped":1`)
}
