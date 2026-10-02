package ignorefiles

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

type tc struct {
	rel   string
	isDir bool
	want  bool
}

func check(t *testing.T, m *Matcher, cases []tc) {
	t.Helper()
	for _, c := range cases {
		got := m.Excluded(filepath.Join(m.Root(), filepath.FromSlash(c.rel)), c.isDir)
		if got != c.want {
			t.Errorf("Excluded(%q, dir=%v) = %v, want %v", c.rel, c.isDir, got, c.want)
		}
	}
}

func TestGitignoreSyntax(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, GitIgnore), `# comment
*.log
!keep.log
build/
/rootonly.go
docs/*.md
**/gen/**
a/**/z.go
\#hash.go
llama-cpp-*/
`)
	m := New(root, nil)
	check(t, m, []tc{
		{"x.log", false, true},
		{"sub/x.log", false, true},
		{"keep.log", false, false}, // later negation wins
		{"build", true, true},
		{"build", false, false}, // trailing slash: dirs only
		{"src/build", true, true},
		{"src/build/main.go", false, true}, // via ancestor
		{"rootonly.go", false, true},
		{"sub/rootonly.go", false, false}, // leading slash anchors
		{"docs/readme.md", false, true},
		{"docs/deep/readme.md", false, false}, // * does not cross /
		{"x/gen/y/z.go", false, true},
		{"a/z.go", false, true},
		{"a/b/c/z.go", false, true},
		{"#hash.go", false, true},
		{"llama-cpp-tq3", true, true},
		{"llama-cpp-tq3/src/ggml.go", false, true},
		{"main.go", false, false},
	})
}

func TestNestedGitignoreIsScopedAndDeeperWins(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, GitIgnore), "*.gen.go\n")
	writeFile(t, filepath.Join(root, "pkg", GitIgnore), "/local.go\n!keep.gen.go\n")
	m := New(root, nil)
	check(t, m, []tc{
		{"local.go", false, false},    // pkg/.gitignore does not apply at root
		{"pkg/local.go", false, true}, // anchored to pkg/
		{"pkg/sub/local.go", false, false},
		{"pkg/a.gen.go", false, true},     // root rule still applies below
		{"pkg/keep.gen.go", false, false}, // deeper negation overrides root
		{"keep.gen.go", false, true},
	})
}

func TestStignoreSyntax(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "more.stignore"), "included-dir\n")
	writeFile(t, filepath.Join(root, SyncthingIgnore), `// Syncthing comment
!important.log
*.log
(?i)CaseDir
(?d)/restore
openviking_workspace/resources
#include more.stignore
`)
	m := New(root, nil)
	check(t, m, []tc{
		{"important.log", false, false}, // first match wins in .stignore
		{"a/important.log", false, false},
		{"other.log", false, true},
		{"casedir", true, true},
		{"x/CASEDIR/f.go", false, true},
		{"restore", true, true},
		{"restore/main.tf", false, true},
		{"sub/restore", true, false}, // "/" anchors to the folder root
		{"openviking_workspace/resources/ast-context-cache/main.go", false, true},
		{"deep/openviking_workspace/resources", true, true}, // unanchored matches any depth
		{"included-dir/x.go", false, true},
		{"main.go", false, false},
	})
}

func TestAstignoreOverridesGitignore(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, GitIgnore), "generated/\n")
	writeFile(t, filepath.Join(root, AstIgnore), "!generated/\nterraform/\n")
	writeFile(t, filepath.Join(root, SyncthingIgnore), "usb-recovery\n")
	m := New(root, nil)
	check(t, m, []tc{
		{"generated", true, false}, // .astignore re-includes a gitignored dir
		{"generated/api.go", false, false},
		{"terraform/main.tf", false, true},
		{"usb-recovery/x.sh", false, true},
	})
}

func TestExtraPatternsTakePrecedence(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, AstIgnore), "keepme/\n")
	m := New(root, []string{"llama-cpp-tq-tom/", "/terraform", "!keepme/", "*.bak.go", ""})
	check(t, m, []tc{
		{"llama-cpp-tq-tom", true, true},
		{"llama-cpp-tq-tom/ggml/x.go", false, true},
		{"terraform/main.tf", false, true},
		{"x/terraform/main.tf", false, false},
		{"keepme/x.go", false, false}, // per-project negation beats .astignore
		{"a/b/c.bak.go", false, true},
		{"main.go", false, false},
	})
}

func TestMatchEntryIgnoresAncestorsButExcludedDoesNot(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, GitIgnore), "/vendored/\n")
	m := New(root, nil)
	file := filepath.Join(root, "vendored", "lib", "x.go")
	if m.MatchEntry(file, false) {
		t.Fatal("MatchEntry should only evaluate the entry itself")
	}
	if !m.Excluded(file, false) {
		t.Fatal("Excluded should honor an ignored ancestor directory")
	}
	if m.Excluded(root, true) || m.Excluded(filepath.Dir(root), true) {
		t.Fatal("root and paths outside root are never excluded")
	}
}

func TestIsIgnoreFileName(t *testing.T) {
	for _, n := range []string{".gitignore", ".astignore", ".stignore"} {
		if !IsIgnoreFileName(n) {
			t.Errorf("%s should be an ignore file", n)
		}
	}
	if IsIgnoreFileName("main.go") {
		t.Error("main.go is not an ignore file")
	}
}
