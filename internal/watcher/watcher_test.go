package watcher

import (
	"testing"
)

func TestNormalizeProjectPath(t *testing.T) {
	dir := t.TempDir()
	got := NormalizeProjectPath(dir)
	if got == "" {
		t.Fatal("expected non-empty path")
	}
	if got != NormalizeProjectPath(dir+"/") {
		t.Fatalf("trailing slash mismatch: %q vs %q", got, NormalizeProjectPath(dir+"/"))
	}
}
