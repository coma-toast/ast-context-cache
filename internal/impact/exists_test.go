package impact

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
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
