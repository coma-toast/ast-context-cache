package ignorepatterns

import (
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestMatchNodeModules(t *testing.T) {
	proj := filepath.Join("/tmp", "proj")
	file := filepath.Join(proj, "frontend", "node_modules", "lodash", "index.js")
	if !Match(file, proj, DefaultGlobs) {
		t.Fatal("node_modules should match default globs")
	}
}

func TestListUsesDefaultsWhenUnset(t *testing.T) {
	dbtest.Init(t)
	_ = db.SetSetting(settingKey, "[]")
	InvalidateCache()
	got := List()
	if len(got) == 0 {
		t.Fatal("expected default globs")
	}
	found := false
	for _, p := range got {
		if p == "**/node_modules/**" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("defaults missing node_modules: %v", got)
	}
}

func TestEnsureDefaultsPersists(t *testing.T) {
	dbtest.Init(t)
	EnsureDefaults()
	raw := db.GetSetting(settingKey, "")
	if raw == "" || raw == "[]" {
		t.Fatalf("expected persisted defaults, got %q", raw)
	}
}

func TestMatchDirOnlyPrunesDirPatterns(t *testing.T) {
	proj := filepath.Join("/tmp", "p")
	globs := []string{"**/gen/**", "gen*", "**/foo", "*.pb.go"}
	if !MatchDir(filepath.Join(proj, "a", "gen"), proj, globs) {
		t.Error("**/gen/** should prune a/gen: every file below it matches")
	}
	// These match the directory name but not the files inside it, so pruning
	// would change which files are skipped.
	for _, d := range []string{"generated", "foo", "x.pb.go"} {
		if MatchDir(filepath.Join(proj, d), proj, globs) {
			t.Errorf("%s must not be pruned by a non-/** glob", d)
		}
	}
	if !MatchDir(filepath.Join(proj, "web", "node_modules"), proj, DefaultGlobs) {
		t.Error("default node_modules glob should prune")
	}
}
