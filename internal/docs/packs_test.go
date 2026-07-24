package docs

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestStarterPackShape(t *testing.T) {
	if len(StarterPack) < 3 {
		t.Fatalf("expected at least 3 starter sources, got %d", len(StarterPack))
	}
	seen := map[string]bool{}
	for _, s := range StarterPack {
		if s.Name == "" || s.URL == "" || s.Type == "" {
			t.Fatalf("incomplete entry: %+v", s)
		}
		if seen[s.Name] {
			t.Fatalf("duplicate name %q", s.Name)
		}
		seen[s.Name] = true
	}
}

func TestInstallStarterPack(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	added, results := InstallStarterPack()
	if added != len(StarterPack) {
		t.Fatalf("added=%d want %d (results=%+v)", added, len(StarterPack), results)
	}
	if len(results) != len(StarterPack) {
		t.Fatalf("results len=%d want %d", len(results), len(StarterPack))
	}
	sources, err := ListSources()
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) < len(StarterPack) {
		t.Fatalf("DB sources=%d want >= %d", len(sources), len(StarterPack))
	}

	// Idempotent: second install upserts the same rows.
	added2, _ := InstallStarterPack()
	if added2 != len(StarterPack) {
		t.Fatalf("second install added=%d want %d", added2, len(StarterPack))
	}
	sources2, err := ListSources()
	if err != nil {
		t.Fatal(err)
	}
	if len(sources2) != len(sources) {
		t.Fatalf("second install grew sources %d → %d", len(sources), len(sources2))
	}
}
