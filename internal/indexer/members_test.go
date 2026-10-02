package indexer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

type indexedSymbol struct {
	name, kind, fqn, skeleton string
	start, end                int
}

// indexSource writes src to dir/name, indexes it, and returns its symbols keyed
// by qualified name (fqn without the file basename).
func indexSource(t *testing.T, name, src string) (string, map[string]indexedSymbol) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	project := t.TempDir()
	file := filepath.Join(project, name)
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := IndexFile(file, project); err != nil {
		t.Fatal(err)
	}
	rows, err := db.IndexDB.Query(`SELECT name, kind, COALESCE(fqn,''), COALESCE(skeleton,''), start_line, end_line FROM symbols WHERE file = ? AND project_path = ?`, file, project)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	out := map[string]indexedSymbol{}
	for rows.Next() {
		var s indexedSymbol
		if err := rows.Scan(&s.name, &s.kind, &s.fqn, &s.skeleton, &s.start, &s.end); err != nil {
			t.Fatal(err)
		}
		out[db.QualifiedName(s.fqn, file, s.name)] = s
	}
	return file, out
}

func wantSymbol(t *testing.T, syms map[string]indexedSymbol, qualified, kind string, start int) indexedSymbol {
	t.Helper()
	s, ok := syms[qualified]
	if !ok {
		keys := make([]string, 0, len(syms))
		for k := range syms {
			keys = append(keys, k)
		}
		t.Fatalf("no symbol %q; indexed: %v", qualified, keys)
	}
	if s.kind != kind || s.start != start {
		t.Fatalf("%s: kind=%s start=%d, want kind=%s start=%d", qualified, s.kind, s.start, kind, start)
	}
	return s
}

const llamaCppPy = `import os


def load_model(path):
    return path


class LlamaCppClient(BaseClient):
    """Talks to a llama.cpp server."""

    default_port = 8080

    def __init__(self, host: str,
                 port: int = 8080,
                 timeout: float = 30.0) -> None:
        self.host = host

    def _list_local_models(self, model_dirs: list[str],
                           recursive: bool = True):
        return []

    @staticmethod
    def load_model(name):
        """Load a model by name."""
        return name

    async def unload(self):
        pass

    class Config:
        def to_litellm_params(self):
            return {}
`

func TestIndexFileIndexesPythonMethodsAsSymbols(t *testing.T) {
	_, syms := indexSource(t, "llamacpp.py", llamaCppPy)

	wantSymbol(t, syms, "load_model", "function", 4)
	wantSymbol(t, syms, "LlamaCppClient", "class", 8)
	wantSymbol(t, syms, "LlamaCppClient.__init__", "method", 13)
	wantSymbol(t, syms, "LlamaCppClient._list_local_models", "method", 18)
	// A decorated method's range starts at its decorator, like top-level ones.
	lm := wantSymbol(t, syms, "LlamaCppClient.load_model", "method", 22)
	if lm.name != "load_model" || lm.end != 25 {
		t.Fatalf("load_model method: name=%q end=%d", lm.name, lm.end)
	}
	if lm.fqn != "llamacpp.py.LlamaCppClient.load_model" {
		t.Fatalf("fqn=%q", lm.fqn)
	}
	wantSymbol(t, syms, "LlamaCppClient.unload", "method", 27)
	wantSymbol(t, syms, "LlamaCppClient.Config", "class", 30)
	wantSymbol(t, syms, "LlamaCppClient.Config.to_litellm_params", "method", 31)
	if len(syms) != 8 {
		t.Fatalf("got %d symbols, want 8: %v", len(syms), syms)
	}

	initSig := "def __init__(self, host: str,\n             port: int = 8080,\n             timeout: float = 30.0) -> None:"
	if got := syms["LlamaCppClient.__init__"].skeleton; got != initSig {
		t.Fatalf("__init__ skeleton:\n%s\nwant:\n%s", got, initSig)
	}
	if got := lm.skeleton; got != "@staticmethod\ndef load_model(name):\n    \"\"\"Load a model by name.\"\"\"" {
		t.Fatalf("load_model skeleton:\n%s", got)
	}
	class := syms["LlamaCppClient"].skeleton
	for _, want := range []string{
		"    def __init__(self, host: str,\n                 port: int = 8080,\n                 timeout: float = 30.0) -> None:",
		"    def _list_local_models(self, model_dirs: list[str],\n                           recursive: bool = True):",
		"    async def unload(self):",
		"    class Config:",
	} {
		if !strings.Contains(class, want) {
			t.Fatalf("class skeleton missing %q:\n%s", want, class)
		}
	}
}

func TestIndexFileIndexesTypeScriptClassesAndMethods(t *testing.T) {
	src := `export class ModelStore extends Base {
  private cache: Map<string, Model> = new Map();
  constructor(private readonly api: Api) { super(); }
  async load(
    id: string,
    opts?: { force: boolean },
  ): Promise<Model> {
    return this.api.get(id);
  }
  get size() { return this.cache.size; }
  #evict() {}
}

export interface Model { id: string }
export type ModelId = string;
abstract class Repo { abstract find(id: string): Model; save() {} }
export function helper<T>(
  value: T,
): T {
  return value;
}
`
	_, syms := indexSource(t, "store.ts", src)
	wantSymbol(t, syms, "ModelStore", "class", 1)
	wantSymbol(t, syms, "ModelStore.constructor", "method", 3)
	load := wantSymbol(t, syms, "ModelStore.load", "method", 4)
	if want := "async load(\n  id: string,\n  opts?: { force: boolean },\n): Promise<Model>"; load.skeleton != want {
		t.Fatalf("load skeleton:\n%s\nwant:\n%s", load.skeleton, want)
	}
	wantSymbol(t, syms, "ModelStore.size", "method", 10)
	wantSymbol(t, syms, "ModelStore.#evict", "method", 11)
	wantSymbol(t, syms, "Model", "interface", 14)
	wantSymbol(t, syms, "ModelId", "type", 15)
	wantSymbol(t, syms, "Repo", "class", 16)
	if find := wantSymbol(t, syms, "Repo.find", "method", 16); find.skeleton != "abstract find(id: string): Model" {
		t.Fatalf("find skeleton=%q", find.skeleton)
	}
	if save := wantSymbol(t, syms, "Repo.save", "method", 16); save.skeleton != "save()" {
		t.Fatalf("save skeleton=%q (single-line class members must be cut to their own span)", save.skeleton)
	}
	if h := wantSymbol(t, syms, "helper", "function", 17); h.skeleton != "export function helper<T>(\n  value: T,\n): T" {
		t.Fatalf("helper skeleton=%q", h.skeleton)
	}
}

func TestIndexFileIndexesJavaScriptMethods(t *testing.T) {
	src := "class Foo { constructor() {} static bar(a, b) { return a } }\nexport class Baz {\n  run() {}\n}\n"
	_, syms := indexSource(t, "foo.js", src)
	wantSymbol(t, syms, "Foo", "class", 1)
	wantSymbol(t, syms, "Foo.constructor", "method", 1)
	if bar := wantSymbol(t, syms, "Foo.bar", "method", 1); bar.skeleton != "static bar(a, b)" {
		t.Fatalf("bar skeleton=%q", bar.skeleton)
	}
	wantSymbol(t, syms, "Baz", "class", 2)
	wantSymbol(t, syms, "Baz.run", "method", 3)
}

func TestIndexFileQualifiesGoMethodsByReceiver(t *testing.T) {
	src := "package p\n\ntype Server[T any] struct{}\n\nfunc (s *Server[T]) Handle(\n\tctx context.Context,\n\treq *Request,\n) (interface{}, error) {\n\treturn nil, nil\n}\n\nfunc Free(a int) {}\n"
	_, syms := indexSource(t, "server.go", src)
	h := wantSymbol(t, syms, "Server.Handle", "method", 5)
	if h.name != "Handle" {
		t.Fatalf("name=%q", h.name)
	}
	if want := "func (s *Server[T]) Handle(\n\tctx context.Context,\n\treq *Request,\n) (interface{}, error)"; h.skeleton != want {
		t.Fatalf("Handle skeleton:\n%s\nwant:\n%s", h.skeleton, want)
	}
	if f := wantSymbol(t, syms, "Free", "function", 12); f.skeleton != "func Free(a int)" {
		t.Fatalf("Free skeleton=%q", f.skeleton)
	}
}

func TestParseSymbolsIncludesMethods(t *testing.T) {
	got := ParseSymbols([]byte(llamaCppPy), "python")
	names := map[string]string{}
	for _, s := range got {
		names[s.Name] = s.Kind
	}
	for name, kind := range map[string]string{"LlamaCppClient": "class", "__init__": "method", "unload": "method", "to_litellm_params": "method"} {
		if names[name] != kind {
			t.Fatalf("ParseSymbols missing %s (%s): %v", name, kind, got)
		}
	}
}

func TestExtractSkeletonKeepsWrappedSignatures(t *testing.T) {
	cases := []struct{ lang, kind, src, want string }{
		{"python", "function", "def f(a,\n      b=')'):  # c\n    return a", "def f(a,\n      b=')'):"},
		{"python", "function", "def g(x: dict[str, int] = {'k': 1},\n      cb=lambda v: v) -> int:\n    pass", "def g(x: dict[str, int] = {'k': 1},\n      cb=lambda v: v) -> int:"},
		{"python", "class", "class A(\n    Base,\n):\n    def m(\n        self,\n    ):\n        pass\n    x = 1", "class A(\n    Base,\n):\n    def m(\n        self,\n    ):\n    x = 1"},
		{"go", "function", "func F(m map[string]interface{}) interface{} {\n\treturn m\n}", "func F(m map[string]interface{}) interface{}"},
		{"typescript", "function", "function f(cb: () => void, n = 1) {\n}", "function f(cb: () => void, n = 1)"},
		{"typescript", "class", "class A {\n  private m: Map<string, number> = new Map();\n  foo(a) {\n  }\n  bar(\n    x: number,\n  ): void {\n  }\n}", "class A {\n  private m: Map<string, number> = new Map();\n  foo(a)\n  bar(\n    x: number,\n  ): void\n}"},
	}
	for _, c := range cases {
		if got := ExtractSkeleton(c.src, c.lang, c.kind); got != c.want {
			t.Errorf("%s %s:\n%s\nwant:\n%s", c.lang, c.kind, got, c.want)
		}
	}
}

func TestGetIndexedFilesMarksOlderParserStale(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	project := t.TempDir()
	py := filepath.Join(project, "a.py")
	sh := filepath.Join(project, "a.sh")
	os.WriteFile(py, []byte("def f():\n    pass\n"), 0o644)
	os.WriteFile(sh, []byte("f() { :; }\n"), 0o644)
	for _, f := range []string{py, sh} {
		if _, _, _, err := IndexFile(f, project); err != nil {
			t.Fatal(err)
		}
	}
	if got := db.GetIndexedFiles(project); got[py].IsZero() || got[sh].IsZero() {
		t.Fatalf("freshly indexed files must not be stale: %v", got)
	}
	// Rows written before parser_version existed carry 0.
	db.IndexDB.Exec(`UPDATE indexed_files SET parser_version = 0 WHERE project_path = ?`, project)
	got := db.GetIndexedFiles(project)
	if _, ok := got[py]; !ok || !got[py].IsZero() {
		t.Fatalf("python file indexed by an older parser should map to zero time, got %v", got[py])
	}
	if got[sh].IsZero() {
		t.Fatal("bash extraction did not change; its file should not be re-indexed")
	}
}

func TestReuseFileSkipsSiblingIndexedByOlderParser(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	sibling := filepath.Join(root, "alpha", "repo")
	fresh := filepath.Join(root, "bravo", "repo")
	os.MkdirAll(sibling, 0o755)
	os.MkdirAll(fresh, 0o755)
	src := "class A:\n    def m(self):\n        pass\n"
	sibFile := filepath.Join(sibling, "a.py")
	newFile := filepath.Join(fresh, "a.py")
	os.WriteFile(sibFile, []byte(src), 0o644)
	os.WriteFile(newFile, []byte(src), 0o644)
	if _, _, _, err := IndexFile(sibFile, sibling); err != nil {
		t.Fatal(err)
	}
	db.IndexDB.Exec(`UPDATE indexed_files SET parser_version = 0 WHERE project_path = ?`, sibling)
	if _, ok := ReuseFile(newFile, fresh, &ReuseSource{ProjectPath: sibling}); ok {
		t.Fatal("rows from an older parser must be re-parsed, not copied")
	}
	db.IndexDB.Exec(`UPDATE indexed_files SET parser_version = ? WHERE project_path = ?`, ParserVersion(sibFile), sibling)
	if n, ok := ReuseFile(newFile, fresh, &ReuseSource{ProjectPath: sibling}); !ok || n != 2 {
		t.Fatalf("current-parser rows should be reused: n=%d ok=%v", n, ok)
	}
}
