package impact

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

type existsOut struct {
	Exists    bool       `json:"exists"`
	Locations []Location `json:"locations"`
	Scope     []string   `json:"checked_scope"`
	Error     string     `json:"error"`
}

func TestHandleCheckSymbolExists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	project := filepath.Join(home, "repo")
	file := filepath.Join(project, "page.ts")
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('SaveButtonId','variable',?,7,7,?)`, file, project)

	var got existsOut
	if err := json.Unmarshal([]byte(HandleCheckSymbolExists(map[string]interface{}{"symbol": "savebuttonid"}, project)), &got); err != nil {
		t.Fatal(err)
	}
	if got.Error != "" {
		t.Fatalf("error: %s", got.Error)
	}
	if !got.Exists || len(got.Locations) != 1 {
		t.Fatalf("got=%+v want one location", got)
	}
	loc := got.Locations[0]
	if loc.File != "page.ts" || loc.Line != 7 || loc.Kind != "variable" {
		t.Fatalf("location=%+v", loc)
	}
	if len(got.Scope) != 1 || got.Scope[0] != project {
		t.Fatalf("checked_scope=%v want [%s]", got.Scope, project)
	}

	var missing existsOut
	json.Unmarshal([]byte(HandleCheckSymbolExists(map[string]interface{}{"symbol": "clickSafe"}, project)), &missing)
	if missing.Exists || len(missing.Locations) != 0 {
		t.Fatalf("renamed symbol should not exist: %+v", missing)
	}
}

func TestHandleCheckSymbolExistsValidatesArgs(t *testing.T) {
	if out := HandleCheckSymbolExists(map[string]interface{}{"symbol": "x"}, ""); out != `{"error": "project_path required"}` {
		t.Fatalf("out=%s", out)
	}
	if out := HandleCheckSymbolExists(map[string]interface{}{}, "/tmp/x"); out != `{"error": "symbol required"}` {
		t.Fatalf("out=%s", out)
	}
}

func TestHandleCheckSymbolExistsFindsMethods(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	project := filepath.Join(home, "repo")
	os.MkdirAll(filepath.Join(project, "clients"), 0o755)
	client := filepath.Join(project, "clients", "llamacpp.py")
	os.WriteFile(client, []byte("class LlamaCppClient:\n    def __init__(self, host: str,\n                 port: int = 8080):\n        self.host = host\n\n    def load_model(self, name):\n        return name\n"), 0o644)
	other := filepath.Join(project, "backend.py")
	os.WriteFile(other, []byte("def load_model(path):\n    return path\n"), 0o644)
	sync := filepath.Join(project, "litellm_sync.py")
	os.WriteFile(sync, []byte("import json\n\n\nclass ModelData:\n    name: str\n\n    def to_litellm_params(self):\n        return {}\n"), 0o644)
	for _, f := range []string{client, other, sync} {
		if _, _, _, err := indexer.IndexFile(f, project); err != nil {
			t.Fatal(err)
		}
	}

	check := func(symbol string) existsOut {
		t.Helper()
		var got existsOut
		if err := json.Unmarshal([]byte(HandleCheckSymbolExists(map[string]interface{}{"symbol": symbol}, project)), &got); err != nil {
			t.Fatal(err)
		}
		if got.Error != "" {
			t.Fatalf("%s: error %s", symbol, got.Error)
		}
		return got
	}

	bare := check("load_model")
	if !bare.Exists || len(bare.Locations) != 2 {
		t.Fatalf("bare load_model should find the module function and the method: %+v", bare.Locations)
	}
	want := map[string]Location{
		"backend.py":          {File: "backend.py", Line: 1, Kind: "function"},
		"clients/llamacpp.py": {File: "clients/llamacpp.py", Line: 6, Kind: "method", QualifiedName: "LlamaCppClient.load_model"},
	}
	for _, loc := range bare.Locations {
		if want[loc.File] != loc {
			t.Fatalf("location %+v, want %+v", loc, want[loc.File])
		}
	}

	for _, q := range []string{"LlamaCppClient.load_model", "llamacppclient.LOAD_MODEL", "llamacpp.py.LlamaCppClient.load_model"} {
		got := check(q)
		if !got.Exists || len(got.Locations) != 1 || got.Locations[0] != want["clients/llamacpp.py"] {
			t.Fatalf("%s: %+v", q, got.Locations)
		}
	}
	// A method whose name exists only as a member (field report: ModelData.to_litellm_params).
	wantParams := Location{File: "litellm_sync.py", Line: 7, Kind: "method", QualifiedName: "ModelData.to_litellm_params"}
	for _, q := range []string{"to_litellm_params", "ModelData.to_litellm_params"} {
		if got := check(q); !got.Exists || len(got.Locations) != 1 || got.Locations[0] != wantParams {
			t.Fatalf("%s: %+v, want %+v", q, got.Locations, wantParams)
		}
	}
	if got := check("LlamaCppClient.__init__"); !got.Exists || got.Locations[0].Line != 2 {
		t.Fatalf("__init__: %+v", got.Locations)
	}
	for _, missing := range []string{"OtherClient.load_model", "LlamaCppClient.unload", "Client.load_model"} {
		if got := check(missing); got.Exists {
			t.Fatalf("%s should not exist: %+v", missing, got.Locations)
		}
	}

	// Plaintext rows store fqn "<path>#plaintext": not a member, so no qualified_name.
	logFile := filepath.Join(project, "app.log")
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"app.log", "plaintext", logFile, 1, 1, "started", logFile+"#plaintext", project); err != nil {
		t.Fatal(err)
	}
	if got := check("app.log"); !got.Exists || len(got.Locations) != 1 || got.Locations[0].QualifiedName != "" {
		t.Fatalf("app.log: %+v", got.Locations)
	}

	g, err := Graph("LlamaCppClient.load_model", project, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.DefinedIn) != 1 || g.DefinedIn[0] != "clients/llamacpp.py" {
		t.Fatalf("impact defined_in=%v", g.DefinedIn)
	}
}
