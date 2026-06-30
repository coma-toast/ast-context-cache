package projectmeta

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func testExcludeDB(t *testing.T) {
	t.Helper()
	home := os.Getenv("HOME")
	if home == "" {
		home = t.TempDir()
		t.Setenv("HOME", home)
	}
	cacheDir := filepath.Join(home, ".astcache-test")
	os.MkdirAll(cacheDir, 0755)
	t.Setenv("DB_PATH", filepath.Join(cacheDir, "usage.db"))
	db.Close()
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(db.Close)
}

func TestIsExcludedBasenameAndPrefix(t *testing.T) {
	testExcludeDB(t)
	_ = db.SetSetting(excludeSettingKey, `["basename:outputs","/tmp/excluded-repo"]`)
	InvalidateExcludeCache()
	if !IsExcluded("/any/where/outputs") {
		t.Fatal("basename:outputs should match")
	}
	if !IsExcluded("/tmp/excluded-repo") {
		t.Fatal("exact path should match")
	}
	if !IsExcluded("/tmp/excluded-repo/sub") {
		t.Fatal("prefix path should match")
	}
	if IsExcluded("/tmp/other-repo") {
		t.Fatal("unlisted repo should not match")
	}
}

func TestDiscoverPathsSkipsExcluded(t *testing.T) {
	home := t.TempDir()
	gitRoot := filepath.Join(home, "git", "keep")
	excluded := filepath.Join(home, "git", "skip")
	os.MkdirAll(filepath.Join(gitRoot, ".git"), 0755)
	os.MkdirAll(filepath.Join(excluded, ".git"), 0755)
	t.Setenv("HOME", home)
	testExcludeDB(t)
	_ = db.SetSetting(excludeSettingKey, `["basename:skip"]`)
	InvalidateExcludeCache()
	paths := DiscoverPaths()
	for _, p := range paths {
		if p == filepath.Clean(excluded) {
			t.Fatalf("excluded path discovered: %v", paths)
		}
	}
	found := false
	for _, p := range paths {
		if p == filepath.Clean(gitRoot) {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected keep repo in %v", paths)
	}
}

func TestExcludeJSONForSettings(t *testing.T) {
	got := ExcludeJSONForSettings(`["/foo"]`)
	if got == "" || got == "[]" {
		t.Fatalf("got %q", got)
	}
	if ExcludeJSONForSettings("") != "[]" {
		t.Fatal("empty should be []")
	}
}
