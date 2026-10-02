package watcher

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/impact"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func writeGoFile(t *testing.T, path, fn string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\n\nfunc "+fn+"() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
}

func rowCount(t *testing.T, q string, args ...interface{}) int {
	t.Helper()
	var n int
	if err := db.IndexDB.QueryRow(q, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// Catch-up runs when a watcher starts, so it is the only chance to notice files
// deleted while ast-mcp was down or the watcher was idle-stopped.
func TestCatchUpPurgesFilesDeletedWhileNotWatching(t *testing.T) {
	prevHook := PostIndexHook
	removedCh := make(chan string, 16)
	PostIndexHook = func(f, _ string, removed bool) {
		if removed {
			removedCh <- f
		}
	}
	t.Cleanup(func() { PostIndexHook = prevHook })

	proj := NormalizeProjectPath(t.TempDir())
	// catchUp stops for a project the watcher doesn't know (one deleted while
	// it waited), so register it the way StartWatcher would.
	RegisterKnownProject(proj)
	t.Cleanup(func() { DeleteWatcher(proj) })
	cleanupWatchers(t)
	keep := filepath.Join(proj, "keep.go")
	tracked := filepath.Join(proj, "llm-benchmark", "dashboard.go")
	untracked := filepath.Join(proj, "restore", "lxc_containers.go")
	realFile := filepath.Join(proj, "litellm_sync.go")
	link := filepath.Join(proj, "litellm-sync.go")
	for f, fn := range map[string]string{keep: "Keep", tracked: "Dashboard", untracked: "Lxc", realFile: "LoadConfig"} {
		writeGoFile(t, f, fn)
		if _, _, _, err := indexer.IndexFile(f, proj); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink("litellm_sync.go", link); err != nil {
		t.Fatal(err)
	}
	// Rows a pre-dedup build wrote for the symlink as if it were its own file.
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('LoadConfig', 'function', ?, 3, 3, ?)`, link, proj); err != nil {
		t.Fatal(err)
	}
	db.UpsertIndexedFile(link, proj, time.Now().Add(time.Hour))
	// Vectors for the tracked file: the old catch-up removed symbols/edges only.
	if _, err := db.IndexDB.Exec(`INSERT INTO vectors (symbol_id, content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (0, 'h1', x'00', 'code', ?, 'Dashboard', 'function', ?)`, tracked, proj); err != nil {
		t.Fatal(err)
	}
	// The untracked file lost its indexed_files row; its symbols remain.
	db.DeleteIndexedFile(untracked, proj)
	for _, f := range []string{tracked, untracked} {
		if err := os.Remove(f); err != nil {
			t.Fatal(err)
		}
	}

	catchUp(proj)

	for _, f := range []string{tracked, untracked, link} {
		for _, q := range []string{
			`SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`,
			`SELECT COUNT(*) FROM edges WHERE source_file = ? AND project_path = ?`,
			`SELECT COUNT(*) FROM vectors WHERE source_file = ? AND project_path = ?`,
			`SELECT COUNT(*) FROM indexed_files WHERE file = ? AND project_path = ?`,
		} {
			if n := rowCount(t, q, f, proj); n != 0 {
				t.Errorf("%s: %q returned %d", filepath.Base(f), q, n)
			}
		}
	}
	for _, f := range []string{keep, realFile} {
		if n := rowCount(t, `SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`, f, proj); n != 1 {
			t.Errorf("%s: symbols=%d want 1", filepath.Base(f), n)
		}
	}
	if n := rowCount(t, `SELECT COUNT(*) FROM symbols WHERE name = 'LoadConfig' AND project_path = ?`, proj); n != 1 {
		t.Errorf("LoadConfig indexed %d times, want 1", n)
	}
	got := map[string]bool{}
	deadline := time.After(5 * time.Second)
	for len(got) < 3 {
		select {
		case f := <-removedCh:
			got[f] = true
		case <-deadline:
			t.Fatalf("PostIndexHook(removed) calls: %v, want tracked, untracked and link", got)
		}
	}
}

func TestCatchUpRemovalsKeepsExistingFilesAfterAbortedWalk(t *testing.T) {
	proj := t.TempDir()
	exists := filepath.Join(proj, "exists.go")
	writeGoFile(t, exists, "E")
	gone := filepath.Join(proj, "gone.go")
	indexed := map[string]time.Time{exists: time.Now(), gone: time.Now()}
	got := catchUpRemovals(proj, indexed, map[string]bool{}, false)
	if len(got) != 1 || got[0] != gone {
		t.Fatalf("aborted walk removals = %v, want only %s", got, gone)
	}
	got = catchUpRemovals(proj, indexed, map[string]bool{}, true)
	if len(got) != 2 {
		t.Fatalf("complete walk removals = %v, want both tracked unseen files", got)
	}
}

// unitEmbedder maps every text to the same vector, so any stored vector is a
// perfect match for any query.
type unitEmbedder struct{}

func unitVec() []float32 {
	v := make([]float32, search.VectorDims)
	v[0] = 1
	return v
}

func (unitEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = unitVec()
	}
	return out, nil
}

func (unitEmbedder) EmbedSingle(string) ([]float32, error) { return unitVec(), nil }

// toolView is what an agent sees for a project through the real read paths:
// check_symbol_exists, get_impact_graph and hybrid search (get_context_capsule).
type toolView struct {
	existsFiles []string // check_symbol_exists(symbol).locations[].file
	impactFiles []string // get_impact_graph(importTarget).impacted_by[].file
	searchFiles []string // HybridSearch(symbol) result files (BM25 + vectors)
}

func viewFor(t *testing.T, proj, symbol, importTarget string) toolView {
	t.Helper()
	var v toolView
	var exists struct {
		Locations []struct {
			File string `json:"file"`
		} `json:"locations"`
	}
	if err := json.Unmarshal([]byte(impact.HandleCheckSymbolExists(map[string]interface{}{"symbol": symbol}, proj)), &exists); err != nil {
		t.Fatal(err)
	}
	for _, l := range exists.Locations {
		v.existsFiles = append(v.existsFiles, l.File)
	}
	var graph struct {
		ImpactedBy []struct {
			File string `json:"file"`
		} `json:"impacted_by"`
	}
	if err := json.Unmarshal([]byte(impact.HandleImpactGraph(map[string]interface{}{"symbol": importTarget}, proj)), &graph); err != nil {
		t.Fatal(err)
	}
	for _, e := range graph.ImpactedBy {
		v.impactFiles = append(v.impactFiles, e.File)
	}
	results, _ := search.HybridSearch(symbol, proj, unitEmbedder{}, 20, nil)
	for _, r := range results {
		if f, _ := r.Data["file"].(string); f != "" {
			v.searchFiles = append(v.searchFiles, db.RelPath(f, proj))
		}
	}
	return v
}

func countOf(files []string, want string) int {
	n := 0
	for _, f := range files {
		if f == want {
			n++
		}
	}
	return n
}

// Field report issues #3 and #5 end to end, through the tool read paths rather
// than table counts: after catch-up, a file deleted while nothing was watching
// is gone from check_symbol_exists, get_impact_graph and search, and a symlinked
// file's symbols are reported once, under the real path.
func TestCatchUpDeletedAndSymlinkedFilesInToolResults(t *testing.T) {
	// Catch-up alone must clean up; main's hook (vector cache delete) is not wired here.
	prevHook := PostIndexHook
	PostIndexHook = nil
	t.Cleanup(func() { PostIndexHook = prevHook })

	proj := NormalizeProjectPath(t.TempDir())
	// catchUp stops for a project the watcher doesn't know (one deleted while
	// it waited), so register it the way StartWatcher would.
	RegisterKnownProject(proj)
	t.Cleanup(func() { DeleteWatcher(proj) })
	cleanupWatchers(t)
	const goneRel = "llm-benchmark/model_registry.go"
	gone := filepath.Join(proj, goneRel)
	if err := os.MkdirAll(filepath.Dir(gone), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(gone, []byte("package p\n\nimport \"example.com/hostsdict\"\n\nfunc BuildHostsDict() { hostsdict.New() }\n"), 0644); err != nil {
		t.Fatal(err)
	}
	realFile := filepath.Join(proj, "litellm_sync.go")
	writeGoFile(t, realFile, "EffortLookup")
	for _, f := range []string{gone, realFile} {
		if _, _, _, err := indexer.IndexFile(f, proj); err != nil {
			t.Fatal(err)
		}
		if err := indexer.EmbedFileSymbols(unitEmbedder{}, f, proj); err != nil {
			t.Fatal(err)
		}
	}
	// Symlink rows as a pre-dedup build wrote them (link indexed as its own file).
	link := filepath.Join(proj, "litellm-sync.go")
	if err := os.Symlink("litellm_sync.go", link); err != nil {
		t.Fatal(err)
	}
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('EffortLookup', 'function', ?, 3, 3, ?)`, link, proj); err != nil {
		t.Fatal(err)
	}
	db.UpsertIndexedFile(link, proj, time.Now().Add(time.Hour))
	// Like the configSync files: rows with no indexed_files entry, which the old
	// tracked-files-only catch-up never reconciled against the disk.
	db.DeleteIndexedFile(gone, proj)

	// Preconditions, so the assertions below are meaningful.
	before := viewFor(t, proj, "BuildHostsDict", "hostsdict")
	if countOf(before.existsFiles, goneRel) != 1 || countOf(before.impactFiles, goneRel) != 1 || countOf(before.searchFiles, goneRel) == 0 {
		t.Fatalf("setup: deleted-to-be file not visible before catch-up: %+v", before)
	}
	if n := countOf(viewFor(t, proj, "EffortLookup", "x").existsFiles, "litellm-sync.go"); n != 1 {
		t.Fatalf("setup: symlink rows not visible before catch-up (%d)", n)
	}

	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	catchUp(proj)

	after := viewFor(t, proj, "BuildHostsDict", "hostsdict")
	if n := countOf(after.existsFiles, goneRel); n != 0 {
		t.Errorf("check_symbol_exists still reports deleted file: %v", after.existsFiles)
	}
	if n := countOf(after.impactFiles, goneRel); n != 0 {
		t.Errorf("get_impact_graph still reports deleted file: %v", after.impactFiles)
	}
	if n := countOf(after.searchFiles, goneRel); n != 0 {
		t.Errorf("search still returns deleted file: %v", after.searchFiles)
	}

	sym := viewFor(t, proj, "EffortLookup", "x")
	if len(sym.existsFiles) != 1 || sym.existsFiles[0] != "litellm_sync.go" {
		t.Errorf("check_symbol_exists(EffortLookup) locations = %v, want only litellm_sync.go", sym.existsFiles)
	}
	if countOf(sym.searchFiles, "litellm-sync.go") != 0 || countOf(sym.searchFiles, "litellm_sync.go") == 0 {
		t.Errorf("search(EffortLookup) files = %v, want the real path only", sym.searchFiles)
	}
}
