package watcher

import (
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
)

func TestMatchWatcherIgnore(t *testing.T) {
	proj := filepath.Join("/tmp", "proj")
	cases := []struct {
		name     string
		file     string
		patterns []string
		want     bool
	}{
		{"empty patterns", filepath.Join(proj, "a.go"), nil, false},
		{"exact file", filepath.Join(proj, "gen", "x.go"), []string{"gen/x.go"}, true},
		{"dir prefix", filepath.Join(proj, "dist", "a.js"), []string{"dist"}, true},
		{"star suffix", filepath.Join(proj, "pkg", "foo.pb.go"), []string{"*.pb.go"}, true},
		{"star path", filepath.Join(proj, "a", "b", "c.go"), []string{"a/*/c.go"}, true},
		{"doublestar suffix", filepath.Join(proj, "x", "y", "gen", "z.go"), []string{"**/gen/z.go"}, true},
		{"doublestar tree", filepath.Join(proj, "out", "a.txt"), []string{"out/**"}, true},
		{"node_modules tree", filepath.Join(proj, "frontend", "node_modules", "lodash", "index.js"), []string{"**/node_modules/**"}, true},
		{"no match", filepath.Join(proj, "src", "main.go"), []string{"dist"}, false},
		{"escape parent", filepath.Join("/other", "x.go"), []string{"*.go"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ignorepatterns.Match(tc.file, proj, tc.patterns)
			if got != tc.want {
				t.Fatalf("MatchWatcherIgnore(%q, %q, %v) = %v, want %v", tc.file, proj, tc.patterns, got, tc.want)
			}
		})
	}
}

func TestMatchOneGlobBase(t *testing.T) {
	if !ignorepatterns.Match(filepath.Join("/tmp", "p", "internal", "foo.pb.go"), filepath.Join("/tmp", "p"), []string{"*.pb.go"}) {
		t.Fatal("basename *.pb.go should match")
	}
}
