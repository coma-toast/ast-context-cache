package indexer

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	sitter "github.com/smacker/go-tree-sitter"
)

func parseImports(t *testing.T, lang, src string) []importRef {
	t.Helper()
	p := sitter.NewParser()
	p.SetLanguage(getSitterLanguage(lang))
	tree, err := p.ParseCtx(context.Background(), nil, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	return collectImports(tree.RootNode(), []byte(src), lang)
}

func TestCollectImportsPythonFunctionLocal(t *testing.T) {
	src := `import os, a.b as ab
from .x import (y as z, w)
from .. import q
from m import *
from __future__ import annotations

class LlamaCppClient:
    def load_model(self, key):
        if key:
            from model_manager.switch import switch_local_model
            return switch_local_model(key)

def top():
    try:
        from model_manager.mm_config import HOSTS
    except ImportError:
        import json
    print(HOSTS)
`
	got := parseImports(t, "python", src)
	want := []importRef{
		{Module: "os"},
		{Module: "a.b"},
		{Module: ".x", Names: []string{"y", "w"}},
		{Module: "..", Names: []string{"q"}},
		{Module: "m", Names: []string{"*"}},
		{Module: "model_manager.switch", Names: []string{"switch_local_model"}, Scope: "LlamaCppClient.load_model"},
		{Module: "model_manager.mm_config", Names: []string{"HOSTS"}, Scope: "top"},
		{Module: "json", Scope: "top"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("imports:\n got %+v\nwant %+v", got, want)
	}
}

func TestCollectImportsJSNestedAndDynamic(t *testing.T) {
	src := `import D, {a as b, c} from './m'
import * as ns from 'n'
import 'side-effect'
import type {T} from './t'
import x = require('r')
export {e} from './e'
export * from './s'
export function outer() { return require('./inner') }
const lazy = async () => { const m = await import('./dyn') }
class K { meth() { require('in-method') } }
`
	got := parseImports(t, "typescript", src)
	want := []importRef{
		{Module: "./m", Names: []string{"*", "a", "c"}},
		{Module: "n", Names: []string{"*"}},
		{Module: "side-effect"},
		{Module: "./t", Names: []string{"T"}},
		{Module: "r"},
		{Module: "./e", Names: []string{"e"}},
		{Module: "./s", Names: []string{"*"}},
		{Module: "./inner", Scope: "outer"},
		{Module: "./dyn", Scope: "lazy"},
		{Module: "in-method", Scope: "K.meth"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("imports:\n got %+v\nwant %+v", got, want)
	}
}

func TestCollectImportsBashSourceInFunction(t *testing.T) {
	src := "source ./lib.sh\nsetup() {\n  if true; then\n    . \"$DIR/common.sh\"\n  fi\n}\n"
	got := parseImports(t, "bash", src)
	want := []importRef{{Module: "./lib.sh"}, {Module: "$DIR/common.sh", Scope: "setup"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("imports:\n got %+v\nwant %+v", got, want)
	}
}

func TestIndexFileRecordsFunctionLocalImportEdges(t *testing.T) {
	project := t.TempDir()
	file := filepath.Join(project, "clients", "llamacpp.py")
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	src := "import os\n\nclass LlamaCppClient:\n    def load_model(self, key):\n        from model_manager.switch import switch_local_model\n        return switch_local_model(key)\n"
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := IndexFile(file, project); err != nil {
		t.Fatal(err)
	}
	rows, err := db.IndexDB.Query(`SELECT COALESCE(source_symbol,''), target, kind FROM edges WHERE source_file = ? AND project_path = ?`, file, project)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var got []string
	for rows.Next() {
		var sym, target, kind string
		if err := rows.Scan(&sym, &target, &kind); err != nil {
			t.Fatal(err)
		}
		got = append(got, strings.Join([]string{kind, target, sym}, "|"))
	}
	sort.Strings(got)
	want := []string{
		"import_names|model_manager.switch::switch_local_model|LlamaCppClient.load_model",
		"import|model_manager.switch|LlamaCppClient.load_model",
		"import|os|",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("edges:\n got %q\nwant %q", got, want)
	}
}
