package context

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const widgetsGo = `package widgets

func WidgetAlpha() int { return 1 }

func WidgetBeta() int { return 2 }
`

// indexedWidgets opens a fresh database and indexes one Go file into a new project.
// No t.Parallel: the db pools, candidate cache and session store are package globals.
func indexedWidgets(t *testing.T) (project, file string) {
	t.Helper()
	dbtest.Init(t)
	project = t.TempDir()
	file = filepath.Join(project, "widgets.go")
	writeAndIndex(t, file, project, widgetsGo)
	return project, file
}

func writeAndIndex(t *testing.T, file, project, src string) {
	t.Helper()
	require.NoError(t, os.WriteFile(file, []byte(src), 0o644))
	_, _, _, err := indexer.IndexFile(file, project)
	require.NoError(t, err)
}

type capsuleReply struct {
	Results []struct {
		Name string `json:"name"`
	} `json:"results"`
	Deduped  int  `json:"deduped"`
	CacheHit bool `json:"cache_hit"`
}

func capsule(t *testing.T, project, sessionID string) capsuleReply {
	t.Helper()
	args := map[string]interface{}{"query": "Widget", "mode": "skeleton"}
	if sessionID != "" {
		args["session_id"] = sessionID
	}
	r := HandleGetContextWithMeta(args, project)
	var out capsuleReply
	require.NoError(t, json.Unmarshal([]byte(r.JSON), &out), r.JSON)
	assert.Equal(t, r.CacheHit, out.CacheHit)
	return out
}

func names(r capsuleReply) []string {
	var out []string
	for _, x := range r.Results {
		out = append(out, x.Name)
	}
	return out
}

// AC23 + AC25: sessions share cached candidates, and each session's own dedup
// applies on its very next call, without waiting for the write buffer.
func TestCapsuleSharesCandidatesAcrossSessionsWithPerSessionDedup(t *testing.T) {
	project, _ := indexedWidgets(t)
	a := capsule(t, project, t.Name()+"-a")
	require.ElementsMatch(t, []string{"WidgetAlpha", "WidgetBeta"}, names(a))
	assert.False(t, a.CacheHit)

	b := capsule(t, project, t.Name()+"-b")
	assert.True(t, b.CacheHit, "a second session is served from the shared cache")
	assert.ElementsMatch(t, names(a), names(b), "another session's returns don't dedup this one")
	assert.Zero(t, b.Deduped)

	a2 := capsule(t, project, t.Name()+"-a")
	assert.True(t, a2.CacheHit)
	assert.Empty(t, a2.Results, "symbols returned to a session are deduped on its next call")
	assert.Equal(t, 2, a2.Deduped)

	none := capsule(t, project, "")
	assert.True(t, none.CacheHit, "calls without a session share the cache too")
	assert.ElementsMatch(t, names(a), names(none))
}

// AC24: a reindex of a file in the project invalidates cached candidates.
func TestCapsuleReindexInvalidatesCache(t *testing.T) {
	project, file := indexedWidgets(t)
	first := capsule(t, project, "")
	require.Len(t, first.Results, 2)
	assert.True(t, capsule(t, project, "").CacheHit)

	writeAndIndex(t, file, project, widgetsGo+"\nfunc WidgetGamma() int { return 3 }\n")
	after := capsule(t, project, "")
	assert.False(t, after.CacheHit, "the commit cleared the project's entries")
	assert.Contains(t, names(after), "WidgetGamma")

	require.NoError(t, indexer.PurgeFile(file, project))
	assert.False(t, capsule(t, project, "").CacheHit, "a purge clears them too")
}

func TestPackScoredResultsSkipsDuplicateWithinList(t *testing.T) {
	project, file := indexedWidgets(t)
	hit := func() search.ScoredResult {
		return search.ScoredResult{Score: 1, Data: map[string]interface{}{"name": "WidgetAlpha", "kind": "function", "file": file, "start_line": 3, "end_line": 3}}
	}
	sid := t.Name()
	results, savings, entry := PackScoredResults([]search.ScoredResult{hit(), hit()}, 10, project, "skeleton", sid, 4000)
	require.Len(t, results, 1)
	assert.Equal(t, 1, savings.DedupedCount)
	assert.Equal(t, 2, entry.HitCount, "the trail counts candidates before dedup")
	assert.Equal(t, []string{"widgets.go#WidgetAlpha@3", "widgets.go#WidgetAlpha@3"}, entry.TopHits)
	_, returned := ReturnedKeys(sid)[SymbolDedupKey(file, "WidgetAlpha", 3)]
	assert.True(t, returned)

	results, _, _ = PackScoredResults([]search.ScoredResult{hit(), hit()}, 10, project, "skeleton", "", 4000)
	assert.Len(t, results, 1, "in-list dedup applies without a session")
}

// BF-14: the capsule honors the caller's limit for the candidate fetch.
func TestCapsuleHonorsLimit(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	src := "package widgets\n"
	for i := range 12 {
		src += fmt.Sprintf("\nfunc Widget%d() int { return %d }\n", i, i)
	}
	writeAndIndex(t, filepath.Join(project, "widgets.go"), project, src)
	r := HandleGetContextWithMeta(map[string]interface{}{"query": "Widget", "mode": "skeleton", "limit": float64(5)}, project)
	var out capsuleReply
	require.NoError(t, json.Unmarshal([]byte(r.JSON), &out), r.JSON)
	assert.NotEmpty(t, out.Results)
	assert.LessOrEqual(t, len(out.Results), 5)
}
