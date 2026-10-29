package indexer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// JD-254: a project root that is itself a symlink to a directory
// (/home/paperclip/configSync -> /syncthing/configSync) indexed 0 files because
// filepath.Walk Lstats its root. The project is keyed under the link's target,
// however the root or the walked directory is spelled.
func TestIndexDirectoryThroughSymlinkedRoot(t *testing.T) {
	for _, tc := range []struct {
		name         string
		dir, project func(link, real string) string
		wantSymbols  int
	}{
		{"link as dir and project", func(l, _ string) string { return l }, func(l, _ string) string { return l }, 2},
		{"link dir, real project", func(l, _ string) string { return l }, func(_, r string) string { return r }, 2},
		{"real dir, link project", func(_, r string) string { return r }, func(l, _ string) string { return l }, 2},
		{"subdir via link, link project", func(l, _ string) string { return filepath.Join(l, "sub") }, func(l, _ string) string { return l }, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testPruneDB(t)
			base := t.TempDir()
			real, link := filepath.Join(base, "real"), filepath.Join(base, "link")
			writeGo(t, filepath.Join(real, "top.go"), "Top")
			writeGo(t, filepath.Join(real, "sub", "nested.go"), "Nested")
			if err := os.Symlink(real, link); err != nil {
				t.Fatal(err)
			}
			if _, err := IndexDirectory(tc.dir(link, real), tc.project(link, real)); err != nil {
				t.Fatal(err)
			}
			n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE project_path = ? AND kind = 'function'`, real)
			if n != tc.wantSymbols {
				t.Fatalf("functions under %s = %d, want %d", real, n, tc.wantSymbols)
			}
			if n := countRows(t, `SELECT COUNT(*) FROM symbols WHERE project_path = ? OR file LIKE ?`, link, link+"/%"); n != 0 {
				t.Fatalf("%d rows keyed under the link spelling, want 0", n)
			}
		})
	}
}

func TestIndexDirectoryRejectsFile(t *testing.T) {
	testPruneDB(t)
	proj := t.TempDir()
	file := filepath.Join(proj, "a.go")
	writeGo(t, file, "A")
	_, err := IndexDirectory(file, proj)
	if !errs.HasCode(err, errs.CodeInvalidInput) {
		t.Fatalf("IndexDirectory(file) err = %v, want invalid input", err)
	}
}
