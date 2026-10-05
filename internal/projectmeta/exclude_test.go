package projectmeta

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestIsExcludedBasenameAndPrefix(t *testing.T) {
	dbtest.Init(t)
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
	home := dbtest.Init(t)
	gitRoot := filepath.Join(home, "git", "keep")
	excluded := filepath.Join(home, "git", "skip")
	os.MkdirAll(filepath.Join(gitRoot, ".git"), 0o755)
	os.MkdirAll(filepath.Join(excluded, ".git"), 0o755)
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
