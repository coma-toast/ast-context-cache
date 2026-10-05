// Package cache holds ranked search candidates shared across sessions.
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	defaultCandidateTTL = 5 * time.Minute
	defaultMaxEntries   = 1000
)

// PipelineMetrics counts candidates at each hybrid-search stage of a cached search.
type PipelineMetrics = search.HybridSearchMetrics

// CandidateCache caches ranked search candidates — never final responses — so that
// per-session dedup, mode, token budget and returned-symbol logging still run on every
// call. Entries are indexed by every project in the queried scope, so a commit to a
// linked child invalidates its parent's entries too.
type CandidateCache struct {
	entries   map[string]*candidateEntry
	byProject map[string]map[string]struct{}
	gens      map[string]uint64 // per-project invalidation counters, see Fetch
	allGen    uint64
	ttl       time.Duration
	max       int

	hits   atomic.Int64
	misses atomic.Int64

	mu sync.Mutex
}

type candidateEntry struct {
	results []search.ScoredResult
	metrics PipelineMetrics
	scope   []string
	created time.Time
}

// Candidates is the process-wide candidate cache.
var Candidates = NewCandidateCache(defaultCandidateTTL, defaultMaxEntries)

func init() {
	// Vectors landing for a project change its hybrid and vector rankings just as a
	// symbol commit does; a hook avoids search importing this package (a cycle).
	search.OnVectorsUpserted = Candidates.ClearProject
}

// NewCandidateCache returns an empty cache whose entries live for ttl, holding at most max.
func NewCandidateCache(ttl time.Duration, max int) *CandidateCache {
	return &CandidateCache{
		entries:   map[string]*candidateEntry{},
		byProject: map[string]map[string]struct{}{},
		gens:      map[string]uint64{},
		ttl:       ttl,
		max:       max,
	}
}

// Key identifies one search: the stage (tool and backend), its inputs, and the
// candidate limit, since a larger limit is not a prefix-compatible superset after fusion.
func Key(stage, query, projectPath, filtersKey, docType string, limit int) string {
	b, _ := json.Marshal([]any{stage, query, projectlinks.NormalizePath(projectPath), filtersKey, docType, limit})
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Get returns a deep copy of the candidates cached under key, so callers may mutate
// them (packing rewrites Data maps) without touching what other sessions see.
func (c *CandidateCache) Get(key string) ([]search.ScoredResult, PipelineMetrics, bool) {
	c.mu.Lock()
	e, ok := c.entries[key]
	if ok && time.Since(e.created) >= c.ttl {
		c.removeLocked(key)
		ok = false
	}
	if !ok {
		c.mu.Unlock()
		c.misses.Add(1)
		return nil, PipelineMetrics{}, false
	}
	results, metrics := e.results, e.metrics
	c.mu.Unlock()
	c.hits.Add(1)
	return cloneResults(results), metrics, true
}

// Set caches a deep copy of results under key for projectPath's resolved scope.
func (c *CandidateCache) Set(key, projectPath string, results []search.ScoredResult, metrics PipelineMetrics) {
	scope := resolveScope(projectPath)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.storeLocked(key, scope, cloneResults(results), metrics)
}

// Fetch returns the candidates cached under key, or runs compute and caches its
// result. hit reports whether the cache answered. A failed compute is not cached, nor
// is a result whose scope was invalidated while compute ran: it may predate the
// commit that cleared it.
func (c *CandidateCache) Fetch(key, projectPath string, compute func() ([]search.ScoredResult, PipelineMetrics, error)) (results []search.ScoredResult, metrics PipelineMetrics, hit bool, err error) {
	if results, metrics, ok := c.Get(key); ok {
		return results, metrics, true, nil
	}
	scope := resolveScope(projectPath)
	c.mu.Lock()
	before := c.genLocked(scope)
	c.mu.Unlock()
	results, metrics, err = compute()
	if err != nil {
		return nil, PipelineMetrics{}, false, err
	}
	stored := cloneResults(results)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.genLocked(scope) == before {
		c.storeLocked(key, scope, stored, metrics)
	}
	return results, metrics, false, nil
}

// ClearProject drops every entry whose scope includes projectPath.
func (c *CandidateCache) ClearProject(projectPath string) {
	p := projectlinks.NormalizePath(projectPath)
	if p == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.gens[p]++
	for key := range c.byProject[p] {
		c.removeLocked(key)
	}
}

// ClearAll drops every entry.
func (c *CandidateCache) ClearAll() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.allGen++
	c.entries = map[string]*candidateEntry{}
	c.byProject = map[string]map[string]struct{}{}
}

// Stats returns lifetime hit and miss counts.
func (c *CandidateCache) Stats() (hits, misses int64) {
	return c.hits.Load(), c.misses.Load()
}

// HitRatio returns hits / (hits + misses), or 0 before any lookup.
func (c *CandidateCache) HitRatio() float64 {
	hits, misses := c.Stats()
	if hits+misses == 0 {
		return 0
	}
	return float64(hits) / float64(hits+misses)
}

// Size returns the capacity and the current entry count.
func (c *CandidateCache) Size() (max, entries int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.max, len(c.entries)
}

func (c *CandidateCache) storeLocked(key string, scope []string, results []search.ScoredResult, metrics PipelineMetrics) {
	c.removeLocked(key)
	if len(c.entries) >= c.max {
		c.evictOldestLocked()
	}
	c.entries[key] = &candidateEntry{results: results, metrics: metrics, scope: scope, created: time.Now()}
	for _, p := range scope {
		keys := c.byProject[p]
		if keys == nil {
			keys = map[string]struct{}{}
			c.byProject[p] = keys
		}
		keys[key] = struct{}{}
	}
}

func (c *CandidateCache) removeLocked(key string) {
	e, ok := c.entries[key]
	if !ok {
		return
	}
	delete(c.entries, key)
	for _, p := range e.scope {
		delete(c.byProject[p], key)
		if len(c.byProject[p]) == 0 {
			delete(c.byProject, p)
		}
	}
}

func (c *CandidateCache) evictOldestLocked() {
	var oldestKey string
	var oldest time.Time
	for k, e := range c.entries {
		if oldestKey == "" || e.created.Before(oldest) {
			oldestKey, oldest = k, e.created
		}
	}
	c.removeLocked(oldestKey)
}

// genLocked sums the counters of scope plus ClearAll's. Each only ever grows, so the
// sum changes exactly when one of them does.
func (c *CandidateCache) genLocked(scope []string) uint64 {
	g := c.allGen
	for _, p := range scope {
		g += c.gens[p]
	}
	return g
}

// resolveScope returns projectPath and its linked children, normalized.
func resolveScope(projectPath string) []string {
	scope := projectlinks.ResolveScope(projectPath)
	if len(scope) == 0 {
		if p := projectlinks.NormalizePath(projectPath); p != "" {
			return []string{p}
		}
	}
	return scope
}

func cloneResults(in []search.ScoredResult) []search.ScoredResult {
	if in == nil {
		return nil
	}
	out := make([]search.ScoredResult, len(in))
	for i, r := range in {
		out[i] = search.ScoredResult{Data: cloneMap(r.Data), Score: r.Score}
	}
	return out
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = cloneValue(v)
	}
	return out
}

func cloneValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		return cloneMap(t)
	case []map[string]any:
		out := make([]map[string]any, len(t))
		for i, m := range t {
			out[i] = cloneMap(m)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, x := range t {
			out[i] = cloneValue(x)
		}
		return out
	case []string:
		return append([]string(nil), t...)
	case []int:
		return append([]int(nil), t...)
	case []float64:
		return append([]float64(nil), t...)
	case []float32:
		return append([]float32(nil), t...)
	default:
		return v
	}
}
