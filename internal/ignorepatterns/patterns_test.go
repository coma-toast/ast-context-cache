package ignorepatterns

import (
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestMatchNodeModules(t *testing.T) {
	proj := filepath.Join("/tmp", "proj")
	file := filepath.Join(proj, "frontend", "node_modules", "lodash", "index.js")
	if !Match(file, proj, DefaultGlobs) {
		t.Fatal("node_modules should match default globs")
	}
}

func TestListUsesDefaultsWhenUnset(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
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
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	EnsureDefaults()
	raw := db.GetSetting(settingKey, "")
	if raw == "" || raw == "[]" {
		t.Fatalf("expected persisted defaults, got %q", raw)
	}
}
