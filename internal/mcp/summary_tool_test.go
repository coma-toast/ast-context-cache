package mcp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

func indexedPythonProject(t *testing.T) (project, file string) {
	t.Helper()
	return indexedPython(t, "llamacpp.py", "class LlamaCppClient:\n    def load_model(self, name):\n        return name\n\n    def unload(self):\n        pass\n")
}

func indexedPython(t *testing.T, name, src string) (project, file string) {
	t.Helper()
	dbtest.Init(t)
	project = t.TempDir()
	file = filepath.Join(project, name)
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := indexer.IndexFile(file, project); err != nil {
		t.Fatal(err)
	}
	return project, file
}

func summaryRows(t *testing.T, project string) int {
	t.Helper()
	var n int
	db.IndexDB.QueryRow(`SELECT COUNT(*) FROM summaries WHERE project_path = ?`, project).Scan(&n)
	return n
}

func TestCacheSummaryRejectsUnindexedSymbol(t *testing.T) {
	project, file := indexedPythonProject(t)
	for _, symbol := range []string{"no_such_symbol", "OtherClient.load_model", "LlamaCppClient.missing"} {
		out := handleCacheSummary(map[string]interface{}{"file": file, "symbol": symbol, "summary": "s"}, project)
		if out["status"] == "cached" || out["error"] == nil {
			t.Fatalf("%s: want an error for an unindexed symbol, got %v", symbol, out)
		}
		data, _ := json.Marshal(out)
		if !resultIsError(data) {
			t.Fatalf("%s: result must surface as an MCP error: %s", symbol, data)
		}
	}
	if n := summaryRows(t, project); n != 0 {
		t.Fatalf("rejected summaries must not be stored, found %d rows", n)
	}
	out := handleCacheSummary(map[string]interface{}{"file": filepath.Join(project, "gone.py"), "summary": "s"}, project)
	if out["error"] == nil {
		t.Fatalf("file-level summary for an unindexed file should fail: %v", out)
	}
	out = handleCacheSummary(map[string]interface{}{"file": file, "symbol": "model", "summary": "s"}, project)
	if got, _ := out["did_you_mean"].([]string); len(got) != 1 || got[0] != "LlamaCppClient.load_model" {
		t.Fatalf("did_you_mean=%v", out["did_you_mean"])
	}
}

func TestCacheSummaryStoresServableMethodSummary(t *testing.T) {
	project, file := indexedPythonProject(t)
	out := handleCacheSummary(map[string]interface{}{"file": "llamacpp.py", "symbol": "LlamaCppClient.load_model", "summary": "Loads a model by name."}, project)
	if out["status"] != "cached" || out["symbol"] != "load_model" || out["qualified_name"] != "LlamaCppClient.load_model" || out["file"] != file {
		t.Fatalf("cache_summary=%v", out)
	}
	// Summary mode must actually serve it (the old handler hashed the summary
	// text, which never matched the code hash LoadSummary checks against).
	if got := summariesByQualifiedName(t, file, project)["LlamaCppClient.load_model"]; got != "Loads a model by name." {
		t.Fatalf("summary mode served %v", got)
	}
	if out := handleCacheSummary(map[string]interface{}{"file": file, "summary": "Client for llama.cpp."}, project); out["status"] != "cached" {
		t.Fatalf("file-level summary for an indexed file: %v", out)
	}
	if got := summariesByQualifiedName(t, file, project)["LlamaCppClient.unload"]; got != "Client for llama.cpp." {
		t.Fatalf("file-level fallback served %v", got)
	}
}

// summariesByQualifiedName returns what get_file_context in summary mode serves
// for each symbol, keyed by qualified name (the bare name for top-level ones).
func summariesByQualifiedName(t *testing.T, file, project string) map[string]interface{} {
	t.Helper()
	var out struct {
		Symbols []map[string]interface{} `json:"symbols"`
	}
	if err := json.Unmarshal([]byte(handleFileContext(file, project, "summary", "", 0)), &out); err != nil {
		t.Fatal(err)
	}
	got := map[string]interface{}{}
	for _, s := range out.Symbols {
		key, _ := s["qualified_name"].(string)
		if key == "" {
			key, _ = s["name"].(string)
		}
		got[key] = s["summary"]
	}
	return got
}

const twoClientsPy = "class LlamaCppClient:\n    def load_model(self, name):\n        return 'llama'\n\n\nclass OMLXClient:\n    def load_model(self, path, quant=None):\n        return 'omlx'\n"

// Summaries were keyed by bare symbol name, so a summary for one class's
// load_model overwrote the other's, and neither was served (the stored code
// hash matched only one of the two same-named rows).
func TestCacheSummaryKeepsSameNamedMethodsApart(t *testing.T) {
	project, file := indexedPython(t, "clients.py", twoClientsPy)
	want := map[string]string{
		"LlamaCppClient.load_model": "Loads via llama.cpp.",
		"OMLXClient.load_model":     "Loads via oMLX.",
	}
	for symbol, summary := range want {
		out := handleCacheSummary(map[string]interface{}{"file": file, "symbol": symbol, "summary": summary}, project)
		if out["status"] != "cached" || out["qualified_name"] != symbol {
			t.Fatalf("%s: %v", symbol, out)
		}
	}
	got := summariesByQualifiedName(t, file, project)
	for symbol, summary := range want {
		if got[symbol] != summary {
			t.Fatalf("%s: summary mode served %v, want %q (all: %v)", symbol, got[symbol], summary, got)
		}
	}
	// A bare name shared by both classes can't say which method it means.
	out := handleCacheSummary(map[string]interface{}{"file": file, "symbol": "load_model", "summary": "?"}, project)
	if out["status"] == "cached" || out["error"] == nil {
		t.Fatalf("ambiguous bare name must be rejected: %v", out)
	}
	if c, _ := out["candidates"].([]string); len(c) != 2 || c[0] != "LlamaCppClient.load_model" || c[1] != "OMLXClient.load_model" {
		t.Fatalf("candidates=%v", out["candidates"])
	}
	if n := summaryRows(t, project); n != 2 {
		t.Fatalf("rejected summary must not be stored: %d rows", n)
	}
}

// retrieve used to re-resolve each hit's lines by bare name, so both
// load_model hits came back with the first one's code.
func TestRetrieveReturnsEachSameNamedMethod(t *testing.T) {
	project, _ := indexedPython(t, "clients.py", twoClientsPy)
	chunks, _, _, _, _ := retrieveCode("load_model", project, 10, false, "skeleton", "", nil, context.PrecisionArgs{Collapse: true})
	got := map[string]string{}
	for _, c := range chunks {
		if c.Kind != "method" {
			continue
		}
		var wire map[string]interface{}
		data, _ := json.Marshal(c)
		json.Unmarshal(data, &wire)
		q, _ := wire["qualified_name"].(string)
		got[q] = c.Content
	}
	want := map[string]string{
		"LlamaCppClient.load_model": "def load_model(self, name):",
		"OMLXClient.load_model":     "def load_model(self, path, quant=None):",
	}
	if len(got) != len(want) {
		t.Fatalf("method chunks %v, want %v", got, want)
	}
	for q, content := range want {
		if got[q] != content {
			t.Fatalf("%s: content %q, want %q (all: %v)", q, got[q], content, got)
		}
	}
}

// Only "<file basename>.<qualified name>" fqns name a member; plaintext rows
// store "<path>#plaintext" and must not surface that as a qualified name.
func TestFileContextOmitsQualifiedNameForNonMemberFqn(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	file := filepath.Join(project, "app.log")
	if err := os.WriteFile(file, []byte("started\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"app.log", "plaintext", file, 1, 1, "started", file+"#plaintext", project); err != nil {
		t.Fatal(err)
	}
	var out struct {
		Symbols []map[string]interface{} `json:"symbols"`
	}
	if err := json.Unmarshal([]byte(handleFileContext(file, project, "skeleton", "", 0)), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Symbols) != 1 {
		t.Fatalf("symbols=%v", out.Symbols)
	}
	if q, ok := out.Symbols[0]["qualified_name"]; ok {
		t.Fatalf("plaintext symbol got qualified_name %v", q)
	}
}

func TestFileContextListsMethodsWithQualifiedNames(t *testing.T) {
	project, file := indexedPythonProject(t)
	var out struct {
		Symbols []map[string]interface{} `json:"symbols"`
	}
	if err := json.Unmarshal([]byte(handleFileContext(file, project, "skeleton", "", 0)), &out); err != nil {
		t.Fatal(err)
	}
	got := map[string]map[string]interface{}{}
	for _, s := range out.Symbols {
		got[s["name"].(string)] = s
	}
	if len(got) != 3 {
		t.Fatalf("want class + 2 methods, got %v", out.Symbols)
	}
	lm := got["load_model"]
	if lm["kind"] != "method" || lm["qualified_name"] != "LlamaCppClient.load_model" || lm["start_line"] != float64(2) || lm["skeleton"] != "def load_model(self, name):" {
		t.Fatalf("load_model=%v", lm)
	}
	if _, ok := got["LlamaCppClient"]["qualified_name"]; ok {
		t.Fatalf("top-level symbols need no qualified_name: %v", got["LlamaCppClient"])
	}
}
