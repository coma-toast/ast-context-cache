package search

import (
	"reflect"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// seedSameNamedMethods indexes two classes in one file that each declare
// load_model, plus a module-level load_model elsewhere, the way the indexer
// stores them: same name and kind, told apart only by fqn.
func seedSameNamedMethods(t *testing.T) (project string, ids map[string]int64) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(db.Close)
	project = "/proj"
	ids = map[string]int64{}
	for _, row := range []struct {
		file, kind, fqn string
		start           int
	}{
		{"/proj/clients.py", "method", "clients.py.LlamaCppClient.load_model", 2},
		{"/proj/clients.py", "method", "clients.py.OMLXClient.load_model", 6},
		{"/proj/backend.py", "function", "backend.py.load_model", 1},
	} {
		res, err := db.IndexDB.Exec(
			`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			"load_model", row.kind, row.file, row.start, row.start+1, "def load_model(self, name):", row.fqn, project)
		if err != nil {
			t.Fatal(err)
		}
		id, _ := res.LastInsertId()
		ids[row.fqn] = id
	}
	return project, ids
}

func unitVector() []float32 {
	v := make([]float32, VectorDims)
	v[0] = 1
	return v
}

func vectorEntries(project string, ids map[string]int64) []VectorEntry {
	var out []VectorEntry
	for fqn, id := range ids {
		file, kind := "/proj/clients.py", "method"
		if fqn == "backend.py.load_model" {
			file, kind = "/proj/backend.py", "function"
		}
		out = append(out, VectorEntry{SymbolID: id, ContentHash: fqn, Vector: unitVector(), DocType: "code",
			SourceFile: file, Name: "load_model", Kind: kind, ProjectPath: project})
	}
	return out
}

// qualifiedByFile maps "file|qualified_name" for each result; a top-level
// symbol must carry no qualified_name.
func qualifiedByFile(t *testing.T, path string, results []ScoredResult) map[string]bool {
	t.Helper()
	got := map[string]bool{}
	for _, r := range results {
		file, _ := r.Data["file"].(string)
		q, hasQ := r.Data["qualified_name"].(string)
		switch r.Data["kind"] {
		case "method":
			if !hasQ {
				t.Fatalf("%s: method result without qualified_name: %v", path, r.Data)
			}
		case "function":
			if hasQ {
				t.Fatalf("%s: top-level result must not carry qualified_name: %v", path, r.Data)
			}
		}
		got[file+"|"+q] = true
	}
	return got
}

var wantQualified = map[string]bool{
	"/proj/clients.py|LlamaCppClient.load_model": true,
	"/proj/clients.py|OMLXClient.load_model":     true,
	"/proj/backend.py|":                          true,
}

func TestSearchResultsQualifyMethodsByClass(t *testing.T) {
	project, ids := seedSameNamedMethods(t)
	for path, results := range map[string][]ScoredResult{
		"bm25":     BM25Search("load_model", project, nil),
		"trigram":  TrigramSearch([]string{"load_model"}, project, nil),
		"fallback": FallbackSearch([]string{"load_model"}, project, nil),
	} {
		if got := qualifiedByFile(t, path, results); !reflect.DeepEqual(got, wantQualified) {
			t.Fatalf("%s: got %v, want %v", path, got, wantQualified)
		}
	}
	vc := &VectorCache{entries: vectorEntries(project, ids), loaded: true, stopIdle: make(chan struct{})}
	if got := qualifiedByFile(t, "vector", vc.Search(unitVector(), project, "", 10, nil)); !reflect.DeepEqual(got, wantQualified) {
		t.Fatalf("vector: got %v, want %v", got, wantQualified)
	}
}

type constEmbedder struct{ v []float32 }

func (e constEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = e.v
	}
	return out, nil
}
func (e constEmbedder) EmbedSingle(string) ([]float32, error) { return e.v, nil }

// RRF fusion keys results by symbol; same-named methods of different classes in
// one file are different symbols and must not collapse into one hit.
func TestHybridSearchKeepsSameNamedMethodsApart(t *testing.T) {
	project, ids := seedSameNamedMethods(t)
	Cache.mu.Lock()
	savedEntries, savedLoaded := Cache.entries, Cache.loaded
	Cache.entries, Cache.loaded, Cache.lastUsed = vectorEntries(project, ids), true, time.Now()
	Cache.mu.Unlock()
	t.Cleanup(func() {
		Cache.mu.Lock()
		Cache.entries, Cache.loaded = savedEntries, savedLoaded
		Cache.mu.Unlock()
	})
	results, metrics := HybridSearch("load_model", project, constEmbedder{unitVector()}, 10, nil)
	if metrics.VectorCandidates == 0 {
		t.Fatal("test needs vector candidates to exercise fusion")
	}
	if got := qualifiedByFile(t, "hybrid", results); !reflect.DeepEqual(got, wantQualified) || len(results) != 3 {
		t.Fatalf("hybrid: %d results %v, want %v", len(results), got, wantQualified)
	}
}
