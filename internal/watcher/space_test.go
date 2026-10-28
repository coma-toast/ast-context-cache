package watcher

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// gitSpace builds a main clone with n Go files committed and a wtg-style space
// (go.work plus one worktree per name, each on its own branch).
func gitSpace(t *testing.T, n int, worktrees ...string) (mainRepo, space string) {
	t.Helper()
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	root := t.TempDir()
	mainRepo = filepath.Join(root, "git", "repo")
	space = filepath.Join(root, "spaces", "feature")
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Skipf("git unavailable: %v: %s", err, out)
		}
	}
	for i := 0; i < n; i++ {
		writeGoFile(t, filepath.Join(mainRepo, fmt.Sprintf("f%02d.go", i)), fmt.Sprintf("F%02d", i))
	}
	git("init", "-q", mainRepo)
	git("-C", mainRepo, "config", "user.email", "test@example.com")
	git("-C", mainRepo, "config", "user.name", "test")
	git("-C", mainRepo, "add", "-A")
	git("-C", mainRepo, "commit", "-q", "-m", "init")
	if err := os.MkdirAll(space, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(space, "go.work"), []byte("go 1.23\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, wt := range worktrees {
		git("-C", mainRepo, "worktree", "add", "-q", "-b", "space-"+wt, filepath.Join(space, wt))
	}
	return NormalizeProjectPath(mainRepo), NormalizeProjectPath(space)
}

// embedProject stores a vector for every symbol of projectPath, standing in for
// the embedder having finished the main clone.
func embedProject(t *testing.T, projectPath string) {
	t.Helper()
	rows, err := db.IndexDB.Query(`SELECT id, name, kind, file, COALESCE(embed_hash,'') FROM symbols WHERE project_path = ?`, projectPath)
	if err != nil {
		t.Fatal(err)
	}
	var entries []search.VectorEntry
	for rows.Next() {
		var e search.VectorEntry
		if err := rows.Scan(&e.SymbolID, &e.Name, &e.Kind, &e.SourceFile, &e.ContentHash); err != nil {
			t.Fatal(err)
		}
		e.Vector, e.DocType, e.ProjectPath = make([]float32, search.VectorDims), "code", projectPath
		e.Vector[0] = 1
		entries = append(entries, e)
	}
	rows.Close()
	if err := search.Cache.Upsert(entries); err != nil {
		t.Fatal(err)
	}
}

// A new space checkout's catch-up copies every unchanged file (rows and vectors)
// from the indexed main clone and only parses and embeds the files that differ.
func TestCatchUpReusesSiblingCheckout(t *testing.T) {
	const files, changed = 40, 3
	mainRepo, space := gitSpace(t, files, "repo")
	wt := filepath.Join(space, "repo")
	if _, err := indexer.IndexDirectory(mainRepo, mainRepo); err != nil {
		t.Fatal(err)
	}
	embedProject(t, mainRepo)
	var want []string
	for i := 0; i < changed; i++ {
		f := filepath.Join(wt, fmt.Sprintf("f%02d.go", i))
		writeGoFile(t, f, fmt.Sprintf("Changed%02d", i))
		want = append(want, f)
	}

	prevHook := PostIndexHook
	var hookMu sync.Mutex
	var embedded []string
	PostIndexHook = func(f, _ string, removed bool) {
		if !removed {
			hookMu.Lock()
			embedded = append(embedded, f)
			hookMu.Unlock()
		}
	}
	t.Cleanup(func() { PostIndexHook = prevHook })
	RegisterKnownProject(wt)
	t.Cleanup(func() { DeleteWatcher(wt) })
	cleanupWatchers(t)

	catchUp(wt)
	bg.Wait()

	if n := rowCount(t, `SELECT COUNT(DISTINCT file) FROM symbols WHERE project_path = ?`, wt); n != files {
		t.Fatalf("checkout indexed %d files, want %d", n, files)
	}
	copiedVectors := rowCount(t, `SELECT COUNT(*) FROM vectors WHERE project_path = ? AND doc_type = 'code'`, wt)
	if copiedVectors != files-changed {
		t.Fatalf("checkout has %d code vectors, want %d copied from the main clone", copiedVectors, files-changed)
	}
	hookMu.Lock()
	sort.Strings(embedded)
	got := append([]string(nil), embedded...)
	hookMu.Unlock()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("embeds enqueued for %v, want only the changed files %v", got, want)
	}
	if n := rowCount(t, `SELECT COUNT(*) FROM symbols WHERE project_path = ? AND name LIKE 'Changed%'`, wt); n != changed {
		t.Fatalf("changed files parsed into %d symbols, want %d", n, changed)
	}
	t.Logf("catch-up of %d files: reused=%d (vectors copied, no embeds) parsed=%d (embeds enqueued=%d); before this change every file was parsed and embedded",
		files, files-changed, changed, len(got))
}

// Starting a watcher on a space root watches each checkout as its own project
// and links them under the root; the root itself gets no watcher or index.
func TestEnsureWatcherOnSpaceRootWatchesCheckouts(t *testing.T) {
	prevHook := PostIndexHook
	PostIndexHook = nil
	t.Cleanup(func() { PostIndexHook = prevHook })
	_, space := gitSpace(t, 2, "repo", "other")
	kids := []string{filepath.Join(space, "other"), filepath.Join(space, "repo")}
	t.Cleanup(func() {
		for _, k := range kids {
			DeleteWatcher(k)
		}
	})
	cleanupWatchers(t)

	EnsureWatcher(space)
	if IsActive(space) {
		t.Fatal("space root must not get a watcher of its own")
	}
	for _, k := range kids {
		if !IsActive(k) {
			t.Fatalf("checkout %s not watched", k)
		}
	}
	links, _ := projectlinks.Links(space)
	if fmt.Sprint(links) != fmt.Sprint(kids) {
		t.Fatalf("links=%v want %v", links, kids)
	}
	for _, k := range kids {
		waitFiles(t, k, 2)
	}
	if n := rowCount(t, `SELECT COUNT(*) FROM symbols WHERE project_path = ?`, space); n != 0 {
		t.Fatalf("space root has %d symbols, want 0", n)
	}

	// A plain folder without checkouts is still watched as one project.
	plain := NormalizeProjectPath(t.TempDir())
	writeGoFile(t, filepath.Join(plain, "a.go"), "A")
	t.Cleanup(func() { DeleteWatcher(plain) })
	EnsureWatcher(plain)
	if !IsActive(plain) {
		t.Fatal("plain folder not watched")
	}
}

// waitFiles waits for a watcher's catch-up to index want files for projectPath.
func waitFiles(t *testing.T, projectPath string, want int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for rowCount(t, `SELECT COUNT(DISTINCT file) FROM symbols WHERE project_path = ?`, projectPath) < want {
		if time.Now().After(deadline) {
			t.Fatalf("%s: catch-up did not index %d files", projectPath, want)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
