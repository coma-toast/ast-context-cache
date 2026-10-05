package cache

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func sampleResults() []search.ScoredResult {
	return []search.ScoredResult{
		{Score: 0.9, Data: map[string]any{"name": "Foo", "file": "/p/a.go", "start_line": 3, "tags": []string{"x"}, "nested": map[string]any{"k": "v"}}},
		{Score: 0.5, Data: map[string]any{"name": "Bar", "file": "/p/b.go", "start_line": 7}},
	}
}

func TestCandidateCacheGetReturnsIsolatedCopies(t *testing.T) {
	c := NewCandidateCache(time.Minute, 10)
	in := sampleResults()
	c.Set("k", "/p", in, PipelineMetrics{BM25Candidates: 2})
	in[0].Data["name"] = "mutated after Set"
	first, metrics, ok := c.Get("k")
	require.True(t, ok)
	assert.Equal(t, 2, metrics.BM25Candidates)
	assert.Equal(t, "Foo", first[0].Data["name"], "Set must store a copy")
	first[0].Data["name"] = "mutated by caller"
	first[0].Data["code"] = "added by ApplyMode"
	first[0].Data["tags"].([]string)[0] = "y"
	first[0].Data["nested"].(map[string]any)["k"] = "changed"
	first[1].Score = 0
	second, _, ok := c.Get("k")
	require.True(t, ok)
	assert.Equal(t, sampleResults(), second, "a caller's mutations must not reach the cache")
}

func TestCandidateCacheExpiresAndEvicts(t *testing.T) {
	c := NewCandidateCache(time.Minute, 2)
	c.Set("a", "/p", sampleResults(), PipelineMetrics{})
	c.Set("b", "/p", sampleResults(), PipelineMetrics{})
	c.Set("c", "/q", sampleResults(), PipelineMetrics{})
	_, _, ok := c.Get("a")
	assert.False(t, ok, "the oldest entry is evicted at capacity")
	max, n := c.Size()
	assert.Equal(t, 2, max)
	assert.Equal(t, 2, n)
	c.Set("a", "/p", sampleResults(), PipelineMetrics{})
	c.ClearProject("/p")
	_, _, ok = c.Get("c")
	assert.True(t, ok, "byProject must not keep keys of evicted entries")
	expired := NewCandidateCache(time.Nanosecond, 2)
	expired.Set("a", "/p", sampleResults(), PipelineMetrics{})
	time.Sleep(time.Millisecond)
	_, _, ok = expired.Get("a")
	assert.False(t, ok)
}

func TestCandidateCacheClearProject(t *testing.T) {
	tests := []struct {
		name   string
		clear  string
		wantP  bool
		wantQ  bool
		viaAll bool
	}{
		{name: "clears only that project", clear: "/p", wantP: false, wantQ: true},
		{name: "path is normalized", clear: "/p/", wantP: false, wantQ: true},
		{name: "unknown project keeps everything", clear: "/elsewhere", wantP: true, wantQ: true},
		{name: "clear all", viaAll: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := NewCandidateCache(time.Minute, 10)
			c.Set(Key("s", "q", "/p", "", "", 30), "/p", sampleResults(), PipelineMetrics{})
			c.Set(Key("s", "q", "/q", "", "", 30), "/q", sampleResults(), PipelineMetrics{})
			if tc.viaAll {
				c.ClearAll()
			} else {
				c.ClearProject(tc.clear)
			}
			_, _, okP := c.Get(Key("s", "q", "/p", "", "", 30))
			_, _, okQ := c.Get(Key("s", "q", "/q", "", "", 30))
			assert.Equal(t, tc.wantP, okP)
			assert.Equal(t, tc.wantQ, okQ)
		})
	}
}

// A parent's search covers its linked children, so a commit to a child must drop
// the parent's cached rankings.
func TestCandidateCacheClearChildInvalidatesParentScope(t *testing.T) {
	dbtest.Init(t)
	parent := t.TempDir()
	child := filepath.Join(parent, "svc")
	require.NoError(t, os.Mkdir(child, 0o755))
	require.NoError(t, projectlinks.CreateLink(parent, child, false))
	c := NewCandidateCache(time.Minute, 10)
	c.Set("parent", parent, sampleResults(), PipelineMetrics{})
	c.Set("child", child, sampleResults(), PipelineMetrics{})
	c.ClearProject(child)
	_, _, ok := c.Get("parent")
	assert.False(t, ok)
	_, _, ok = c.Get("child")
	assert.False(t, ok)
}

func TestCandidateCacheFetch(t *testing.T) {
	c := NewCandidateCache(time.Minute, 10)
	calls := 0
	compute := func() ([]search.ScoredResult, PipelineMetrics, error) {
		calls++
		return sampleResults(), PipelineMetrics{HybridAfterFuse: 2}, nil
	}
	got, _, hit, err := c.Fetch("k", "/p", compute)
	require.NoError(t, err)
	assert.False(t, hit)
	got[0].Data["name"] = "mutated"
	got, m, hit, err := c.Fetch("k", "/p", compute)
	require.NoError(t, err)
	assert.True(t, hit)
	assert.Equal(t, 1, calls)
	assert.Equal(t, 2, m.HybridAfterFuse)
	assert.Equal(t, "Foo", got[0].Data["name"])
	hits, misses := c.Stats()
	assert.Equal(t, int64(1), hits)
	assert.Equal(t, int64(1), misses)
	assert.InDelta(t, 0.5, c.HitRatio(), 1e-9)

	_, _, _, err = c.Fetch("failing", "/p", func() ([]search.ScoredResult, PipelineMetrics, error) {
		return nil, PipelineMetrics{}, errors.New("embed failed")
	})
	require.Error(t, err)
	_, _, ok := c.Get("failing")
	assert.False(t, ok, "a failed search is not cached")

	_, _, _, err = c.Fetch("raced", "/p", func() ([]search.ScoredResult, PipelineMetrics, error) {
		c.ClearProject("/p") // a commit lands while the search runs
		return sampleResults(), PipelineMetrics{}, nil
	})
	require.NoError(t, err)
	_, _, ok = c.Get("raced")
	assert.False(t, ok, "rankings computed across a commit may predate it")
}

func TestCandidateCacheConcurrentAccess(t *testing.T) {
	c := NewCandidateCache(time.Minute, 16)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				key := fmt.Sprintf("k%d", i%24)
				project := fmt.Sprintf("/p%d", i%3)
				switch (g + i) % 5 {
				case 0:
					c.Set(key, project, sampleResults(), PipelineMetrics{})
				case 1:
					if r, _, ok := c.Get(key); ok {
						r[0].Data["name"] = "x"
					}
				case 2:
					c.ClearProject(project)
				case 3:
					c.Stats()
					c.HitRatio()
					c.Size()
				case 4:
					c.Fetch(key, project, func() ([]search.ScoredResult, PipelineMetrics, error) {
						return sampleResults(), PipelineMetrics{}, nil
					})
				}
			}
		}(g)
	}
	wg.Wait()
	_, n := c.Size()
	assert.LessOrEqual(t, n, 16)
}

func TestKeyDistinguishesInputs(t *testing.T) {
	base := Key("capsule", "q", "/p", "", "", 30)
	assert.Equal(t, base, Key("capsule", "q", "/p/", "", "", 30))
	for _, other := range []string{
		Key("retrieve", "q", "/p", "", "", 30),
		Key("capsule", "q2", "/p", "", "", 30),
		Key("capsule", "q", "/r", "", "", 30),
		Key("capsule", "q", "/p", "p:internal", "", 30),
		Key("capsule", "q", "/p", "", "doc", 30),
		Key("capsule", "q", "/p", "", "", 20),
	} {
		assert.NotEqual(t, base, other)
	}
}
