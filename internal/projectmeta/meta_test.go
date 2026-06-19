package projectmeta

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWorkspaceForPath(t *testing.T) {
	home := t.TempDir()
	spaces := filepath.Join(home, "spaces", "nightly", "slapi")
	os.MkdirAll(spaces, 0755)
	t.Setenv("HOME", home)
	got := Enrich(spaces)
	if got.Workspace != "nightly" {
		t.Fatalf("workspace=%q", got.Workspace)
	}
	if got.Label != "slapi · nightly" {
		t.Fatalf("label=%q", got.Label)
	}
	if got.RepoName != "slapi" {
		t.Fatalf("repo=%q", got.RepoName)
	}
}

func TestDiscoverPathsIncludesSpaces(t *testing.T) {
	home := t.TempDir()
	slapi := filepath.Join(home, "spaces", "pipeline", "slapi")
	os.MkdirAll(filepath.Join(slapi, ".git"), 0755)
	t.Setenv("HOME", home)
	paths := DiscoverPaths()
	found := false
	for _, p := range paths {
		if p == filepath.Clean(slapi) {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("discover paths=%v", paths)
	}
}
