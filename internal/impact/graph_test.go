package impact

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

// indexProject writes files under a fresh project in an isolated HOME and runs
// the real indexer over them.
func indexProject(t *testing.T, files map[string]string) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(db.Close)
	project := filepath.Join(home, "proj")
	for name, body := range files {
		p := filepath.Join(project, name)
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for name := range files {
		if _, _, _, err := indexer.IndexFile(filepath.Join(project, name), project); err != nil {
			t.Fatalf("index %s: %v", name, err)
		}
	}
	return project
}

func impactedFiles(res *Result) map[string]Entry {
	out := map[string]Entry{}
	for _, e := range res.ImpactedBy {
		out[e.File] = e
	}
	return out
}

func TestGraphFindsFunctionLocalImportCaller(t *testing.T) {
	project := indexProject(t, map[string]string{
		"model_manager/__init__.py": "",
		"model_manager/switch.py":   "def switch_local_model(name):\n    return True\n",
		"model_manager/clients/llamacpp.py": `import os

class LlamaCppClient:
    def load_model(self, key):
        from model_manager.switch import switch_local_model
        if not switch_local_model(key):
            return False
`,
		"model_manager/cli.py":   "def main():\n    from model_manager import switch_local_model\n    switch_local_model('x')\n",
		"model_manager/other.py": "def f():\n    from model_manager.switch import something_else\n",
	})
	res, err := Graph("switch_local_model", project, false)
	if err != nil {
		t.Fatal(err)
	}
	got := impactedFiles(res)
	e, ok := got["model_manager/clients/llamacpp.py"]
	if !ok {
		t.Fatalf("llamacpp.py missing from impacted_by: %+v", res.ImpactedBy)
	}
	if !reflect.DeepEqual(e.Callers, []string{"LlamaCppClient.load_model"}) {
		t.Fatalf("callers = %v, want [LlamaCppClient.load_model]", e.Callers)
	}
	if c := got["model_manager/cli.py"].Callers; !reflect.DeepEqual(c, []string{"main"}) {
		t.Fatalf("package re-export import: callers = %v, want [main] (entries %+v)", c, res.ImpactedBy)
	}
	if _, ok := got["model_manager/other.py"]; ok {
		t.Fatal("other.py imports a different name from switch.py and cannot reach switch_local_model")
	}
	if !reflect.DeepEqual(res.DefinedIn, []string{"model_manager/switch.py"}) {
		t.Fatalf("defined_in = %v", res.DefinedIn)
	}
}

func TestGraphNamedImportsOfUndeclaredNameIgnoreYAMLKeys(t *testing.T) {
	// HOSTS is served by a PEP 562 module __getattr__, so no code declares it;
	// the only same-named "symbols" are lowercase YAML keys.
	project := indexProject(t, map[string]string{
		"model_manager/mm_config.py": "def load_config():\n    return {}\n\ndef __getattr__(name):\n    if name == 'HOSTS':\n        return {}\n",
		"model_manager/commands/lifecycle.py": `def start(host):
    from model_manager.mm_config import HOSTS
    return HOSTS[host]

def stop(host):
    from model_manager.mm_config import HOSTS, load_config
    return HOSTS[host]
`,
		"model_manager/commands/registry.py": "def reg():\n    from model_manager.mm_config import load_config\n    return load_config()\n",
		"config.yaml":                        "hosts:\n  mbp: {}\n",
		"playbook.yml":                       "- hosts: all\n  tasks: []\n",
		"design-system/eslint.config.js":     "import { defineConfig } from \"eslint/config\";\nexport default defineConfig([]);\n",
	})
	res, err := Graph("HOSTS", project, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.DefinedIn) != 0 {
		t.Fatalf("defined_in = %v, want none (YAML `hosts:` keys are not HOSTS)", res.DefinedIn)
	}
	got := impactedFiles(res)
	e, ok := got["model_manager/commands/lifecycle.py"]
	if !ok || !reflect.DeepEqual(e.Callers, []string{"start", "stop"}) {
		t.Fatalf("lifecycle.py entry = %+v (ok=%v), want callers [start stop]", e, ok)
	}
	if len(got) != 1 {
		t.Fatalf("impacted_by = %+v, want only lifecycle.py", res.ImpactedBy)
	}
}

func TestGraphNoSubstringMatchOnImportTargets(t *testing.T) {
	project := indexProject(t, map[string]string{
		"organize/config.py":             "def load_config():\n    return {}\n",
		"organize/run.py":                "from organize.config import load_config\n",
		"organize/rel.py":                "from .config import load_config\n",
		"design-system/eslint.config.js": "import { defineConfig } from \"eslint/config\";\nexport default defineConfig([]);\n",
		"web/eslint/config.ts":           "export const rules = {}\n",
		"web/app.ts":                     "import { rules } from 'eslint/config'\nimport { other } from './configure'\n",
		"web/configure.ts":               "export const other = 1\n",
	})
	res, err := Graph("load_config", project, false)
	if err != nil {
		t.Fatal(err)
	}
	var files []string
	for f := range impactedFiles(res) {
		files = append(files, f)
	}
	sort.Strings(files)
	if want := []string{"organize/rel.py", "organize/run.py"}; !reflect.DeepEqual(files, want) {
		t.Fatalf("impacted_by files = %v, want %v", files, want)
	}

	// A bare specifier still resolves when a local file spells every segment.
	res, err = Graph("rules", project, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := impactedFiles(res)["web/app.ts"]; !ok || len(res.ImpactedBy) != 1 {
		t.Fatalf("rules impacted_by = %+v, want only web/app.ts", res.ImpactedBy)
	}
}

func TestGraphDefinedInPrefersCodeOverData(t *testing.T) {
	project := indexProject(t, map[string]string{
		"settings.py":      "TIMEOUT = 5\ndef TIMEOUTS():\n    pass\n",
		"values.yaml":      "TIMEOUTS: 5\nreplicas: 2\n",
		"deploy.yaml":      "replicas: 3\n",
		"docs/TIMEOUTS.md": "# TIMEOUTS\n",
	})
	res, err := Graph("TIMEOUTS", project, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(res.DefinedIn, []string{"settings.py"}) {
		t.Fatalf("defined_in = %v, want [settings.py]", res.DefinedIn)
	}
	// With no code declaration, data definitions are still reported.
	res, err = Graph("replicas", project, false)
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(res.DefinedIn)
	if !reflect.DeepEqual(res.DefinedIn, []string{"deploy.yaml", "values.yaml"}) {
		t.Fatalf("defined_in = %v, want both YAML files", res.DefinedIn)
	}
}

func TestResolveModule(t *testing.T) {
	cases := []struct {
		module, src, def string
		want             moduleMatch
	}{
		{"model_manager.mm_config", "/p/mm/model_manager/clients/a.py", "/p/mm/model_manager/mm_config.py", matchFile},
		{"model_manager", "/p/mm/model_manager/cli.py", "/p/mm/model_manager/__init__.py", matchFile},
		{"model_manager", "/p/mm/model_manager/cli.py", "/p/mm/model_manager/switch.py", matchPackage},
		{"mm_config", "/p/a.py", "/p/model_manager/mm_config.py", matchFile},
		{"other.mm_config", "/p/a.py", "/p/model_manager/mm_config.py", matchNone},
		{".config", "/p/organize/run.py", "/p/organize/config.py", matchFile},
		{"..config", "/p/organize/sub/run.py", "/p/organize/config.py", matchFile},
		{".config", "/p/other/run.py", "/p/organize/config.py", matchNone},
		{"eslint/config", "/p/design-system/eslint.config.js", "/p/organize/config.py", matchNone},
		{"eslint/config", "/p/design-system/eslint.config.js", "/p/src/config.ts", matchNone},
		{"eslint/config", "/p/web/app.ts", "/p/web/eslint/config.ts", matchFile},
		{"./page", "/p/tests/spec.ts", "/p/tests/page.ts", matchFile},
		{"./page.js", "/p/tests/spec.ts", "/p/tests/page.ts", matchFile},
		{"../lib", "/p/src/a/b.ts", "/p/src/lib/index.ts", matchFile},
		{"./components", "/p/src/app.tsx", "/p/src/components/button/Button.tsx", matchPackage},
		{"@/lib/api", "/p/src/app.tsx", "/p/src/lib/api.ts", matchFile},
		{"./pagex", "/p/tests/spec.ts", "/p/tests/page.ts", matchNone},
		{"github.com/x/repo/internal/impact", "/r/cmd/main.go", "/r/internal/impact/handler.go", matchFile},
		{"github.com/x/repo/internal/impactx", "/r/cmd/main.go", "/r/internal/impact/handler.go", matchNone},
		{"./lib.sh", "/p/bin/run.sh", "/p/bin/lib.sh", matchFile},
		{"$DIR/lib/common.sh", "/p/bin/run.sh", "/p/bin/lib/common.sh", matchFile},
		{"$DIR/common.sh", "/p/bin/run.sh", "/p/bin/uncommon.sh", matchNone},
		{"./modules/vpc", "/p/main.tf", "/p/modules/vpc/main.tf", matchFile},
	}
	for _, c := range cases {
		if got := resolveModule(c.module, c.src, c.def); got != c.want {
			t.Errorf("resolveModule(%q, %q, %q) = %v, want %v", c.module, c.src, c.def, got, c.want)
		}
	}
}
