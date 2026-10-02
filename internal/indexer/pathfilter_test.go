package indexer

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
)

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// lockedDir creates an unreadable directory: filepath.Walk fails (and
// IndexDirectory returns that error) if it ever descends into it, so a clean
// walk proves the excluded parent was pruned rather than filtered per file.
func lockedDir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(path, 0o755) })
}

func indexedRel(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	for f := range db.GetIndexedFiles(root) {
		rel, _ := filepath.Rel(root, f)
		out = append(out, filepath.ToSlash(rel))
	}
	sort.Strings(out)
	return out
}

func TestIndexDirectoryHonorsIgnoreFilesAndPrunesDirs(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ignorepatterns.InvalidateCache()
	root := t.TempDir()
	src := "package p\n\nfunc F() {}\n"
	writeTestFile(t, filepath.Join(root, "main.go"), src)
	writeTestFile(t, filepath.Join(root, ".gitignore"), "gitignored/\n")
	writeTestFile(t, filepath.Join(root, "gitignored", "a.go"), src)
	writeTestFile(t, filepath.Join(root, "sub", ".gitignore"), "local_only.go\n")
	writeTestFile(t, filepath.Join(root, "sub", "local_only.go"), src)
	writeTestFile(t, filepath.Join(root, "sub", "kept.go"), src)
	writeTestFile(t, filepath.Join(root, ".stignore"), "llama-cpp-tq3\n")
	writeTestFile(t, filepath.Join(root, "llama-cpp-tq3", "x.go"), src)
	writeTestFile(t, filepath.Join(root, ".astignore"), "astignored/\n")
	writeTestFile(t, filepath.Join(root, "astignored", "x.go"), src)
	writeTestFile(t, filepath.Join(root, "perproj", "x.go"), src)
	writeTestFile(t, filepath.Join(root, "out", "x.go"), src) // default global glob **/out/**
	for _, d := range []string{"gitignored", "llama-cpp-tq3", "astignored", "perproj", "out"} {
		lockedDir(t, filepath.Join(root, d, "locked"))
	}
	if err := db.SetProjectIndexExcludes(root, []string{"perproj/"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.SetProjectIndexExcludes(root, nil) })

	if _, err := IndexDirectory(root, root); err != nil {
		t.Fatalf("IndexDirectory walked into an excluded dir: %v", err)
	}
	got := indexedRel(t, root)
	want := []string{"main.go", "sub/kept.go"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("indexed %v, want %v", got, want)
	}
}

func TestIndexDirectorySubdirRespectsAncestorIgnore(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, ".gitignore"), "/vendored/\n")
	writeTestFile(t, filepath.Join(root, "vendored", "lib", "x.go"), "package lib\n\nfunc X() {}\n")
	if _, err := IndexDirectory(filepath.Join(root, "vendored", "lib"), root); err != nil {
		t.Fatal(err)
	}
	if got := indexedRel(t, root); len(got) != 0 {
		t.Fatalf("indexed %v under an ignored ancestor", got)
	}
}

func TestPathFilterExcludedForPurge(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ignorepatterns.InvalidateCache()
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, ".astignore"), "terraform/\n")
	if err := db.SetProjectIndexExcludes(root, []string{"restore/"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.SetProjectIndexExcludes(root, nil) })
	f := NewPathFilter(root)
	cases := map[string]bool{
		"main.go":                     false,
		"terraform/modules/a/b.tf":    true, // .astignore ancestor
		"restore/x.go":                true, // per-project list
		"web/node_modules/x/index.js": true, // global default glob
		".github/workflows/ci.yml":    true, // ShouldSkipDir ancestor
		"web/dist/app.js":             true, // global default glob
	}
	for rel, want := range cases {
		if got := f.Excluded(filepath.Join(root, filepath.FromSlash(rel))); got != want {
			t.Errorf("Excluded(%s) = %v, want %v", rel, got, want)
		}
	}
}

func TestCachedPathFilterInvalidation(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	a := CachedPathFilter(root)
	if CachedPathFilter(root) != a {
		t.Fatal("expected cached filter reuse")
	}
	InvalidatePathFilter(root)
	b := CachedPathFilter(root)
	if b == a {
		t.Fatal("InvalidatePathFilter should force a rebuild")
	}
	InvalidateAllPathFilters()
	if CachedPathFilter(root) == b {
		t.Fatal("InvalidateAllPathFilters should force a rebuild")
	}
}
