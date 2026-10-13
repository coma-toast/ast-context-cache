package search

import (
	"math"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func hit(score float64, file string, line int, name string) ScoredResult {
	return ScoredResult{Score: score, Data: map[string]interface{}{"file": file, "start_line": line, "name": name}}
}

func TestLessScored(t *testing.T) {
	tests := []struct {
		name string
		a, b ScoredResult
		want bool
	}{
		{"higher score first", hit(2, "z.go", 9, "z"), hit(1, "a.go", 1, "a"), true},
		{"lower score last", hit(1, "a.go", 1, "a"), hit(2, "z.go", 9, "z"), false},
		{"tie breaks on file", hit(1, "a.go", 9, "z"), hit(1, "b.go", 1, "a"), true},
		{"tie breaks on start_line", hit(1, "a.go", 1, "z"), hit(1, "a.go", 2, "a"), true},
		{"tie breaks on name", hit(1, "a.go", 1, "a"), hit(1, "a.go", 1, "b"), true},
		{"identical is not less", hit(1, "a.go", 1, "a"), hit(1, "a.go", 1, "a"), false},
		{"missing keys sort first", ScoredResult{Score: 1, Data: map[string]interface{}{}}, hit(1, "a.go", 1, "a"), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, LessScored(tt.a, tt.b))
		})
	}
}

func TestLessVector(t *testing.T) {
	e := func(id int64, file, name string) *VectorEntry {
		return &VectorEntry{ID: id, SourceFile: file, Name: name}
	}
	tests := []struct {
		name       string
		a, b       *VectorEntry
		simA, simB float64
		want       bool
	}{
		{"higher similarity first", e(9, "z.go", "z"), e(1, "a.go", "a"), 0.9, 0.1, true},
		{"tie breaks on source file", e(9, "a.go", "z"), e(1, "b.go", "a"), 0.5, 0.5, true},
		{"tie breaks on name", e(9, "a.go", "a"), e(1, "a.go", "b"), 0.5, 0.5, true},
		{"tie breaks on id", e(1, "a.go", "a"), e(2, "a.go", "a"), 0.5, 0.5, true},
		{"identical is not less", e(1, "a.go", "a"), e(1, "a.go", "a"), 0.5, 0.5, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, LessVector(tt.a, tt.simA, tt.b, tt.simB))
		})
	}
}

// loadEntries swaps the vector cache contents for a test and restores it after.
func loadEntries(t *testing.T, entries []VectorEntry) {
	Cache.mu.Lock()
	Cache.entries, Cache.loaded, Cache.lastUsed = entries, true, time.Now()
	Cache.mu.Unlock()
	t.Cleanup(Cache.Unload)
}

// scaledVec points along the first two axes, so its cosine with unitVec(0) is cos.
func scaledVec(cos float64) []float32 {
	v := make([]float32, VectorDims)
	v[0] = float32(cos)
	v[1] = float32(math.Sqrt(1 - cos*cos))
	return v
}

func unitVec() []float32 {
	v := make([]float32, VectorDims)
	v[0] = 1
	return v
}

func names(results []ScoredResult) []string {
	out := make([]string, len(results))
	for i, r := range results {
		out[i], _ = r.Data["name"].(string)
	}
	return out
}

// TestVectorSearchSortsUnderLimit is AC1: fewer candidates than limit still come
// back best first (the old partial sort only ran when candidates exceeded limit).
func TestVectorSearchSortsUnderLimit(t *testing.T) {
	var entries []VectorEntry
	for i, c := range []struct {
		name string
		cos  float64
	}{{"low", 0.2}, {"high", 0.9}, {"mid", 0.5}} {
		for _, dt := range []string{"code", "doc", "note", "memory"} {
			entries = append(entries, VectorEntry{ID: int64(i), DocType: dt, SourceFile: dt + ":" + c.name, Name: c.name, ProjectPath: "s1", Vector: scaledVec(c.cos)})
		}
	}
	loadEntries(t, entries)
	want := []string{"high", "mid", "low"}
	assert.Equal(t, want, names(Cache.Search(unitVec(), "", "code", 10, nil)))
	assert.Equal(t, want, names(Cache.SearchDoc(unitVec(), 10)))
	assert.Equal(t, want, names(Cache.SearchNote(unitVec(), "s1", 10)))
	assert.Equal(t, want, names(Cache.SearchMemory(unitVec(), "s1", false, 10)))
	assert.Equal(t, want, names(Cache.SearchMemory(unitVec(), "s1", true, 10)))
}

func TestVectorSearchTiesAreDeterministic(t *testing.T) {
	vec := unitVec()
	loadEntries(t, []VectorEntry{
		{ID: 3, DocType: "doc", SourceFile: "doc:1:3", Name: "c", Vector: vec},
		{ID: 1, DocType: "doc", SourceFile: "doc:1:1", Name: "b", Vector: vec},
		{ID: 2, DocType: "doc", SourceFile: "doc:1:1", Name: "a", Vector: vec},
	})
	assert.Equal(t, []string{"a", "b", "c"}, names(Cache.SearchDoc(vec, 10)))
	assert.Equal(t, []string{"a", "b"}, names(Cache.SearchDoc(vec, 2)))
}

// TestSearchMemorySessionScope covers BF-8: with includeSessionless the session
// filter is skipped (the caller's SQL re-select scopes results) and the candidate
// pool grows to max(limit*5, 50); session-only callers keep their own session.
func TestSearchMemorySessionScope(t *testing.T) {
	var entries []VectorEntry
	for i, sess := range []string{"s1", "s2", ""} {
		entries = append(entries, VectorEntry{ID: int64(i), DocType: "memory", SourceFile: "mem:m" + sess, Name: "m" + sess, ProjectPath: sess, Vector: scaledVec(0.5)})
	}
	for i := 0; i < 60; i++ {
		entries = append(entries, VectorEntry{ID: int64(100 + i), DocType: "memory", SourceFile: "mem:x", Name: "x", ProjectPath: "s3", Vector: scaledVec(0.1)})
	}
	loadEntries(t, entries)
	assert.Equal(t, []string{"ms1"}, names(Cache.SearchMemory(unitVec(), "s1", false, 10)))
	got := names(Cache.SearchMemory(unitVec(), "s1", true, 2))
	require.Len(t, got, 50)
	assert.Equal(t, []string{"m", "ms1", "ms2"}, got[:3])
	assert.Len(t, Cache.SearchMemory(unitVec(), "s1", true, 20), 63)
}

type fixedEmbedder struct{ vec []float32 }

func (f fixedEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = f.vec
	}
	return out, nil
}

func (f fixedEmbedder) EmbedSingle(string) ([]float32, error) { return f.vec, nil }

func initIndex(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	require.NoError(t, db.Init())
	t.Cleanup(db.Close)
}

func insertSymbol(t *testing.T, name, file string, line int) {
	_, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		name, "function", file, line, line+1, "func "+name+"()", "pkg."+name, "/proj")
	require.NoError(t, err)
}

// TestHybridSearchTiesAreDeterministic fuses BM25-only and vector-only hits that
// share ranks (so RRF scores tie) and checks every run returns the same order,
// with ties broken on the fused key.
func TestHybridSearchTiesAreDeterministic(t *testing.T) {
	initIndex(t)
	insertSymbol(t, "alpha", "b.go", 1)
	insertSymbol(t, "alphabet", "d.go", 1)
	vec := unitVec()
	loadEntries(t, []VectorEntry{
		{ID: 1, DocType: "code", SourceFile: "a.go", Name: "zeta", Kind: "function", ProjectPath: "/proj", Vector: vec},
		{ID: 2, DocType: "code", SourceFile: "c.go", Name: "eta", Kind: "function", ProjectPath: "/proj", Vector: scaledVec(0.5)},
	})
	first, _ := HybridSearch("alpha", "/proj", fixedEmbedder{vec: vec}, 10, nil)
	require.Len(t, first, 4)
	for i := 0; i < 20; i++ {
		got, _ := HybridSearch("alpha", "/proj", fixedEmbedder{vec: vec}, 10, nil)
		require.Equal(t, names(first), names(got))
	}
	for i := 1; i < len(first); i++ {
		if first[i-1].Score == first[i].Score {
			assert.Less(t, resultKey(first[i-1]), resultKey(first[i]))
		}
	}
	assert.Equal(t, "zeta", names(first)[0], "a.go key sorts before the tied b.go BM25 hit")
}

// The fused hits record which lists they came from; a hit in both lists takes the vector
// list's similarity, and StripFusionKeys restores the BM25 hit's own fields.
func TestHybridSearchRecordsListMembership(t *testing.T) {
	initIndex(t)
	insertSymbol(t, "alpha", "b.go", 1)
	insertSymbol(t, "alphabet", "d.go", 1)
	vec := unitVec()
	loadEntries(t, []VectorEntry{
		{ID: 1, DocType: "code", SourceFile: "b.go", Name: "alpha", Kind: "function", ProjectPath: "/proj", Vector: vec},
		{ID: 2, DocType: "code", SourceFile: "c.go", Name: "eta", Kind: "function", ProjectPath: "/proj", Vector: scaledVec(0.5)},
	})
	got, _ := HybridSearch("alpha", "/proj", fixedEmbedder{vec: vec}, 10, nil)
	byName := map[string]map[string]interface{}{}
	for _, r := range got {
		byName[r.Data["name"].(string)] = r.Data
	}
	require.Len(t, byName, 3)
	both, bm25, vector := byName["alpha"], byName["alphabet"], byName["eta"]
	assert.Equal(t, true, both[KeyInBM25])
	assert.Equal(t, true, both[KeyInVector])
	assert.InDelta(t, 1, both["similarity"], 1e-6)
	assert.Equal(t, true, bm25[KeyInBM25])
	assert.NotContains(t, bm25, KeyInVector)
	assert.NotContains(t, bm25, "similarity")
	assert.NotContains(t, vector, KeyInBM25)
	assert.Equal(t, true, vector[KeyInVector])
	StripFusionKeys(both)
	StripFusionKeys(vector)
	assert.NotContains(t, both, "similarity", "fusion's similarity is dropped from a BM25 hit")
	assert.NotContains(t, both, KeyInBM25)
	assert.NotContains(t, both, KeyInVector)
	assert.Contains(t, vector, "similarity", "a vector-only hit keeps its own similarity")
	assert.NotContains(t, vector, KeyInVector)
}

func TestFallbackSearchTiesAreDeterministic(t *testing.T) {
	initIndex(t)
	insertSymbol(t, "handle", "b.go", 5)
	insertSymbol(t, "handle", "a.go", 9)
	insertSymbol(t, "handle", "a.go", 3)
	insertSymbol(t, "handler", "a.go", 1)
	got := FallbackSearch([]string{"handle"}, "/proj", nil)
	require.Len(t, got, 4)
	var order []string
	for _, r := range got {
		order = append(order, dataString(r.Data, "file")+":"+dataString(r.Data, "name"))
	}
	assert.Equal(t, []string{"a.go:handle", "a.go:handle", "b.go:handle", "a.go:handler"}, order)
	assert.Equal(t, 3, dataInt(got[0].Data, "start_line"))
}
