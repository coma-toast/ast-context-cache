package search

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// TestScopedCountAndSearchResolveScopeOnce guards against the per-entry project_links
// query that made index-health Count and vector Search take ~25-60s on a 237k-vector cache.
func TestScopedCountAndSearchResolveScopeOnce(t *testing.T) {
	dir := t.TempDir()
	prev := db.SetHomeForTest(dir)
	defer prev()
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	parent := filepath.Join(dir, "git")
	child := filepath.Join(parent, "child")
	other := filepath.Join(dir, "other")
	for _, p := range []string{child, other} {
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := projectlinks.CreateLink(parent, child, false); err != nil {
		t.Fatal(err)
	}
	const n = 200000
	zero := make([]float32, VectorDims) // shared backing array keeps 200k entries cheap
	entries := make([]VectorEntry, 0, n+2)
	for i := 0; i < n; i++ {
		pp := child
		if i%2 == 1 {
			pp = other
		}
		entries = append(entries, VectorEntry{ID: int64(i), ContentHash: strconv.Itoa(i), DocType: "code", Vector: zero, SourceFile: "a.go", Name: "f", ProjectPath: pp})
	}
	vec := make([]float32, VectorDims)
	vec[0] = 1
	entries = append(entries,
		VectorEntry{ID: n, ContentHash: "hit", Vector: vec, DocType: "code", SourceFile: "hit.go", Name: "hit", ProjectPath: child},
		VectorEntry{ID: n + 1, ContentHash: "miss", Vector: vec, DocType: "code", SourceFile: "miss.go", Name: "miss", ProjectPath: other},
	)
	Cache.mu.Lock()
	Cache.entries, Cache.loaded, Cache.lastUsed = entries, true, time.Now()
	Cache.mu.Unlock()
	defer Cache.Unload()

	before := projectlinks.LinksQueryCountForTest()
	got := Cache.Count(parent)
	if q := projectlinks.LinksQueryCountForTest() - before; q != 1 {
		t.Fatalf("Count(parent) ran %d project_links queries over %d entries, want 1", q, len(entries))
	}
	if want := n/2 + 1; got != want {
		t.Fatalf("Count(parent)=%d want %d (linked child entries)", got, want)
	}
	before = projectlinks.LinksQueryCountForTest()
	res := Cache.Search(vec, parent, "", 5, nil)
	if q := projectlinks.LinksQueryCountForTest() - before; q != 1 {
		t.Fatalf("Search(parent) ran %d project_links queries over %d entries, want 1", q, len(entries))
	}
	if len(res) == 0 || res[0].Data["name"] != "hit" {
		t.Fatalf("Search(parent) top=%v want hit from linked child", res)
	}
	for _, r := range res {
		if r.Data["name"] == "miss" {
			t.Fatalf("Search(parent) returned out-of-scope entry: %v", r.Data)
		}
	}
}
