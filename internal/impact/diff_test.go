package impact

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// newRepo creates a git repo with one committed file and returns its path.
func newRepo(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	git := func(args ...string) {
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Skipf("git unavailable: %v: %s", err, out)
		}
	}
	git("init", "-q")
	git("config", "user.email", "test@example.com")
	git("config", "user.name", "test")
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	git("add", "-A")
	git("commit", "-q", "-m", "base")
}

func TestHandleDiffImpactFindsDependents(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	project := filepath.Join(home, "repo")
	newRepo(t, project, map[string]string{
		"page.ts": "export function clickSave() {}\n",
		"spec.ts": "import { clickSave } from './page'\n",
	})

	pageFile := filepath.Join(project, "page.ts")
	specFile := filepath.Join(project, "spec.ts")
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('clickSave','function',?,1,1,?)`, pageFile, project)
	db.IndexDB.Exec(`INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, './page', 'import', ?)`, specFile, project)

	git := func(args ...string) {
		cmd := exec.Command("git", append([]string{"-C", project}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	git("branch", "-q", "base-branch")
	if err := os.WriteFile(pageFile, []byte("export function clickSave() { return 1 }\n"), 0644); err != nil {
		t.Fatal(err)
	}
	git("commit", "-qam", "change page")

	out := HandleDiffImpact(map[string]interface{}{"base_ref": "base-branch"}, project)
	var got struct {
		ChangedFiles   []string            `json:"changed_files"`
		SymbolsChanged []string            `json:"symbols_changed"`
		ImpactedBy     map[string][]string `json:"impacted_by"`
		Error          string              `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal %s: %v", out, err)
	}
	if got.Error != "" {
		t.Fatalf("error: %s", got.Error)
	}
	if len(got.ChangedFiles) != 1 || got.ChangedFiles[0] != "page.ts" {
		t.Fatalf("changed_files=%v want [page.ts]", got.ChangedFiles)
	}
	if len(got.SymbolsChanged) != 1 || got.SymbolsChanged[0] != "clickSave" {
		t.Fatalf("symbols_changed=%v want [clickSave]", got.SymbolsChanged)
	}
	dependents := got.ImpactedBy["clickSave"]
	if len(dependents) != 1 || dependents[0] != "spec.ts" {
		t.Fatalf("impacted_by[clickSave]=%v want [spec.ts]", dependents)
	}
}

func TestHandleDiffImpactRequiresProjectPath(t *testing.T) {
	if out := HandleDiffImpact(map[string]interface{}{}, ""); out != `{"error": "project_path required"}` {
		t.Fatalf("out=%s", out)
	}
}

func TestHandleDiffImpactReportsGitFailure(t *testing.T) {
	dir := t.TempDir()
	out := HandleDiffImpact(map[string]interface{}{"base_ref": "nope"}, dir)
	var got map[string]string
	json.Unmarshal([]byte(out), &got)
	if got["error"] == "" {
		t.Fatalf("expected a git error, got %s", out)
	}
}

func TestOriginRepoParsesGitHubRemotes(t *testing.T) {
	home := t.TempDir()
	project := filepath.Join(home, "repo")
	newRepo(t, project, map[string]string{"a.go": "package p\n"})
	for _, url := range []string{
		"git@github.com:slidehq/slapi.git",
		"https://github.com/slidehq/slapi.git",
		"https://github.com/slidehq/slapi",
	} {
		cmd := exec.Command("git", "-C", project, "remote", "add", "origin", url)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("remote add: %v: %s", err, out)
		}
		if got := originRepo(project); got != "slidehq/slapi" {
			t.Fatalf("originRepo(%s)=%q want slidehq/slapi", url, got)
		}
		exec.Command("git", "-C", project, "remote", "remove", "origin").Run()
	}
	if got := originRepo(project); got != "" {
		t.Fatalf("originRepo without remote=%q want empty", got)
	}
}

func TestIntArgAcceptsJSONForms(t *testing.T) {
	if got := intArg(map[string]interface{}{"pr": float64(453)}, "pr"); got != 453 {
		t.Fatalf("float64 pr=%d", got)
	}
	if got := intArg(map[string]interface{}{"pr": "453"}, "pr"); got != 453 {
		t.Fatalf("string pr=%d", got)
	}
	if got := intArg(map[string]interface{}{"pr": "abc"}, "pr"); got != 0 {
		t.Fatalf("bad pr=%d want 0", got)
	}
	if got := intArg(map[string]interface{}{}, "pr"); got != 0 {
		t.Fatalf("missing pr=%d want 0", got)
	}
}

func TestStripPrefixForSubdirProjects(t *testing.T) {
	if rel, ok := stripPrefix("services/api/main.go", "services/api/"); !ok || rel != "main.go" {
		t.Fatalf("rel=%q ok=%v", rel, ok)
	}
	if _, ok := stripPrefix("web/app.ts", "services/api/"); ok {
		t.Fatal("file outside the project prefix should be dropped")
	}
	if rel, ok := stripPrefix("main.go", ""); !ok || rel != "main.go" {
		t.Fatalf("repo-root project: rel=%q ok=%v", rel, ok)
	}
}

func TestHandleDiffImpactPRReportsFetchFailure(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	project := filepath.Join(home, "repo")
	newRepo(t, project, map[string]string{"a.go": "package p\n"})

	// No remote and no such PR: the tool must surface the gh failure, not hang.
	out := HandleDiffImpact(map[string]interface{}{"pr": float64(999999), "repo": "slidehq/does-not-exist-xyz"}, project)
	var got map[string]interface{}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal %s: %v", out, err)
	}
	if got["error"] == nil || got["error"] == "" {
		t.Fatalf("expected a gh error, got %s", out)
	}
}
