package watcher

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/fsnotify/fsnotify"
)

const goSrc = "package p\n\nfunc F() {}\n"

// newExcludeTestProject returns an empty, registered project dir. The package shares one
// db.Init (TestMain); handleFSEvent queues re-index timers, awaited on cleanup.
func newExcludeTestProject(t *testing.T) string {
	t.Helper()
	root := NormalizeProjectPath(t.TempDir())
	// catchUp stops for a project the watcher doesn't know, so register it.
	RegisterKnownProject(root)
	t.Cleanup(func() { DeleteWatcher(root) })
	cleanupWatchers(t)
	return root
}

func writeExcludeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func indexForTest(t *testing.T, file, root string) {
	t.Helper()
	if _, _, _, err := indexer.IndexFile(file, root); err != nil {
		t.Fatal(err)
	}
}

// Files indexed before an exclude existed must be purged by catch-up, and the
// catch-up walk must prune the excluded directory (the unreadable dir inside it
// would abort the walk, leaving zz_kept.go unseen and wrongly purged).
func TestCatchUpPrunesAndPurgesNewlyExcludedDir(t *testing.T) {
	root := newExcludeTestProject(t)
	vendored := filepath.Join(root, "llama-cpp-tq3", "ggml.go")
	kept := filepath.Join(root, "zz_kept.go")
	writeExcludeTestFile(t, vendored, goSrc)
	writeExcludeTestFile(t, kept, goSrc)
	indexForTest(t, vendored, root)
	indexForTest(t, kept, root)

	writeExcludeTestFile(t, filepath.Join(root, ".gitignore"), "llama-cpp-tq3/\n")
	locked := filepath.Join(root, "llama-cpp-tq3", "locked")
	if err := os.MkdirAll(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(locked, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })

	catchUp(root)
	indexed := db.GetIndexedFiles(root)
	if _, ok := indexed[vendored]; ok {
		t.Fatal("gitignored file should be purged by catch-up")
	}
	if _, ok := indexed[kept]; !ok {
		t.Fatal("kept file was purged: catch-up walked into the excluded dir instead of pruning it")
	}
}

func TestPurgeExcludedHonorsPerProjectList(t *testing.T) {
	root := newExcludeTestProject(t)
	dup := filepath.Join(root, "restore", "old", "main.go")
	keep := filepath.Join(root, "main.go")
	writeExcludeTestFile(t, dup, goSrc)
	writeExcludeTestFile(t, keep, goSrc)
	indexForTest(t, dup, root)
	indexForTest(t, keep, root)

	if err := db.SetProjectIndexExcludes(root, []string{"restore/"}); err != nil {
		t.Fatal(err)
	}
	if n := PurgeExcluded(root); n != 1 {
		t.Fatalf("PurgeExcluded removed %d files, want 1", n)
	}
	var syms int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM symbols WHERE file = ? AND project_path = ?`, dup, root).Scan(&syms)
	if syms != 0 {
		t.Fatalf("%d symbols left for purged file", syms)
	}
	if _, ok := db.GetIndexedFiles(root)[keep]; !ok {
		t.Fatal("non-excluded file was purged")
	}
	if n := PurgeExcluded(root); n != 0 {
		t.Fatalf("second purge removed %d, want 0", n)
	}
}

func TestHandleFSEventHonorsIgnoreFiles(t *testing.T) {
	root := newExcludeTestProject(t)
	writeExcludeTestFile(t, filepath.Join(root, ".astignore"), "generated/\n")
	ignoredFile := filepath.Join(root, "generated", "api.go")
	writeExcludeTestFile(t, ignoredFile, goSrc)
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		t.Fatal(err)
	}
	defer fw.Close()
	w := &fsnotifyBackend{w: fw}
	indexer.InvalidatePathFilter(root)

	handleFSEvent(fsnotify.Event{Name: ignoredFile, Op: fsnotify.Write}, root, w)
	debounceMu.Lock()
	_, scheduled := debounceTimers[debounceKey(root, ignoredFile)]
	debounceMu.Unlock()
	if scheduled {
		t.Fatal("re-index scheduled for a file excluded by .astignore")
	}

	ignoredDir := filepath.Join(root, "generated", "nested")
	okDir := filepath.Join(root, "src")
	for _, d := range []string{ignoredDir, okDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		handleFSEvent(fsnotify.Event{Name: d, Op: fsnotify.Create}, root, w)
	}
	watched := map[string]bool{}
	for _, p := range fw.WatchList() {
		watched[p] = true
	}
	if watched[ignoredDir] || !watched[okDir] {
		t.Fatalf("watch list %v: want %s watched and %s not", fw.WatchList(), okDir, ignoredDir)
	}

	// Editing an ignore file drops the cached filter so the next event sees it.
	before := indexer.CachedPathFilter(root)
	writeExcludeTestFile(t, filepath.Join(root, ".gitignore"), "src/\n")
	handleFSEvent(fsnotify.Event{Name: filepath.Join(root, ".gitignore"), Op: fsnotify.Write}, root, w)
	if indexer.CachedPathFilter(root) == before {
		t.Fatal("ignore-file change should invalidate the cached path filter")
	}
	if !indexer.CachedPathFilter(root).IgnoredByFiles(filepath.Join(okDir, "x.go"), false) {
		t.Fatal("new .gitignore rule not applied after invalidation")
	}
	DeleteWatcher(root) // stop the debounced purge timer
}
