package db

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

// writeFile writes a file last modified well before legacyMinIdle, like the months-old
// strays from the field report.
func writeFile(t *testing.T, path string, data string) {
	t.Helper()
	writeFresh(t, path, data)
	old := time.Now().Add(-2 * legacyMinIdle)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}
}

// writeFresh writes a file with a current mtime (as if another process just created it).
func writeFresh(t *testing.T, path string, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func exists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil
}

// Field report #12: zero-byte ast-cache.db / ast.db / astcache.db littered the data dir.
func TestRemoveEmptyLegacyDBs(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("DB_PATH", "")
	dir := filepath.Join(home, ".astcache")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, "ast-cache.db"), "")          // empty: removed
	writeFile(t, filepath.Join(dir, "ast.db"), "SQLite format 3") // has data: kept
	writeFile(t, filepath.Join(dir, "astcache.db"), "")           // empty but has a WAL: kept
	writeFile(t, filepath.Join(dir, "astcache.db-wal"), "frames")
	for _, active := range []string{"index.db", "context.db", "usage.db", "index.db-wal", "index.db-shm", "usage.db-wal"} {
		writeFile(t, filepath.Join(dir, active), "") // active DBs, even empty: never touched
	}

	removed := removeEmptyLegacyDBs(dir)
	if len(removed) != 1 || removed[0] != "ast-cache.db" {
		t.Fatalf("removed=%v want [ast-cache.db]", removed)
	}
	for _, keep := range []string{"ast.db", "astcache.db", "astcache.db-wal", "index.db", "context.db", "usage.db", "index.db-wal", "index.db-shm", "usage.db-wal"} {
		if !exists(filepath.Join(dir, keep)) {
			t.Fatalf("%s was removed", keep)
		}
	}
	if exists(filepath.Join(dir, "ast-cache.db")) {
		t.Fatal("empty ast-cache.db still present")
	}
}

func TestRemoveEmptyLegacyDBsSkipsActiveAndSymlinks(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := t.TempDir()
	// DB_PATH pointing at a legacy name makes that file the active usage DB.
	t.Setenv("DB_PATH", filepath.Join(dir, "ast.db"))
	writeFile(t, filepath.Join(dir, "ast.db"), "")
	target := filepath.Join(t.TempDir(), "elsewhere.db")
	writeFile(t, target, "")
	if err := os.Symlink(target, filepath.Join(dir, "astcache.db")); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, "ast-cache.db"), "")

	removed := removeEmptyLegacyDBs(dir)
	sort.Strings(removed)
	if len(removed) != 1 || removed[0] != "ast-cache.db" {
		t.Fatalf("removed=%v want [ast-cache.db]", removed)
	}
	if !exists(filepath.Join(dir, "ast.db")) || !exists(filepath.Join(dir, "astcache.db")) || !exists(target) {
		t.Fatal("active DB_PATH file or symlink (or its target) was removed")
	}
}

func TestInitRemovesEmptyLegacyDBs(t *testing.T) {
	if IndexDB != nil {
		Close()
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("DB_PATH", "")
	dir := filepath.Join(home, ".astcache")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, "ast.db"), "")
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(Close)
	if exists(filepath.Join(dir, "ast.db")) {
		t.Fatal("Init left the empty legacy ast.db in place")
	}
	if !exists(filepath.Join(dir, "index.db")) {
		t.Fatal("Init did not create index.db")
	}
}

// A zero-byte legacy file that was just created may belong to another process that has
// opened it and not written yet (SQLite writes the header on the first write). It must be
// left alone until it has sat idle for legacyMinIdle.
func TestRemoveEmptyLegacyDBsSkipsRecentlyModified(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("DB_PATH", "")
	dir := t.TempDir()
	writeFresh(t, filepath.Join(dir, "ast.db"), "")
	writeFile(t, filepath.Join(dir, "astcache.db"), "")
	justUnder := time.Now().Add(-legacyMinIdle + time.Minute)
	writeFresh(t, filepath.Join(dir, "ast-cache.db"), "")
	if err := os.Chtimes(filepath.Join(dir, "ast-cache.db"), justUnder, justUnder); err != nil {
		t.Fatal(err)
	}

	removed := removeEmptyLegacyDBs(dir)
	if len(removed) != 1 || removed[0] != "astcache.db" {
		t.Fatalf("removed=%v want [astcache.db] (only the idle one)", removed)
	}
	if !exists(filepath.Join(dir, "ast.db")) || !exists(filepath.Join(dir, "ast-cache.db")) {
		t.Fatal("a recently modified empty legacy file was removed")
	}
}
