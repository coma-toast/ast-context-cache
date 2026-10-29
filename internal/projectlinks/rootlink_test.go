package projectlinks

import (
	"os"
	"path/filepath"
	"testing"
)

// symlinkTree builds base/real (a directory with sub/ and a.go), plus links to it:
// base/link (absolute), base/rel (relative), base/chain -> link, base/file.go ->
// real/a.go and base/dangling.
func symlinkTree(t *testing.T) string {
	t.Helper()
	base := t.TempDir()
	real := filepath.Join(base, "real")
	if err := os.MkdirAll(filepath.Join(real, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(real, "a.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for link, target := range map[string]string{
		"link":     real,
		"rel":      "real",
		"chain":    filepath.Join(base, "link"),
		"file.go":  filepath.Join(real, "a.go"),
		"dangling": filepath.Join(base, "missing"),
	} {
		if err := os.Symlink(target, filepath.Join(base, link)); err != nil {
			t.Fatal(err)
		}
	}
	return base
}

func TestResolveRootLink(t *testing.T) {
	base := symlinkTree(t)
	real := filepath.Join(base, "real")
	for _, tc := range []struct {
		name, in, want string
	}{
		{"real dir unchanged", real, real},
		{"absolute link", filepath.Join(base, "link"), real},
		{"relative link", filepath.Join(base, "rel"), real},
		{"link chain", filepath.Join(base, "chain"), real},
		{"subdir under link is not a leaf link", filepath.Join(base, "link", "sub"), filepath.Join(base, "link", "sub")},
		{"file link unchanged", filepath.Join(base, "file.go"), filepath.Join(base, "file.go")},
		{"dangling link unchanged", filepath.Join(base, "dangling"), filepath.Join(base, "dangling")},
		{"missing path unchanged", filepath.Join(base, "nope"), filepath.Join(base, "nope")},
		{"empty", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveRootLink(tc.in); got != tc.want {
				t.Fatalf("ResolveRootLink(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
	if got := CanonicalProject(filepath.Join(base, "link") + "/"); got != real {
		t.Fatalf("CanonicalProject(link/) = %q, want %q", got, real)
	}
}

func TestRebaseIntoProject(t *testing.T) {
	base := symlinkTree(t)
	real, link := filepath.Join(base, "real"), filepath.Join(base, "link")
	other := t.TempDir()
	for _, tc := range []struct {
		name, path, project, want string
	}{
		{"already under project", filepath.Join(real, "sub"), real, filepath.Join(real, "sub")},
		{"link root onto real project", link, real, real},
		{"link subdir onto real project", filepath.Join(link, "sub"), real, filepath.Join(real, "sub")},
		{"real subdir onto link project", filepath.Join(real, "sub"), link, filepath.Join(link, "sub")},
		{"outside project unchanged", other, real, other},
		{"missing path unchanged", filepath.Join(link, "nope"), real, filepath.Join(link, "nope")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := RebaseIntoProject(tc.path, tc.project); got != tc.want {
				t.Fatalf("RebaseIntoProject(%q, %q) = %q, want %q", tc.path, tc.project, got, tc.want)
			}
		})
	}
}
