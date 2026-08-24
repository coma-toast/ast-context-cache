package impact

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

type deletionOut struct {
	Removed         []string            `json:"removed_symbols"`
	StillReferenced map[string][]string `json:"still_referenced_elsewhere"`
	Safe            []string            `json:"safe_to_delete"`
	Error           string              `json:"error"`
}

func TestHandleCheckDeletionSafety(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	project := filepath.Join(home, "repo")
	base := "export function clickSave() {}\nexport function clickCancel() {}\n"
	newRepo(t, project, map[string]string{
		"page.ts": base,
		"spec.ts": "import { clickSave } from './page'\n",
	})

	pageFile := filepath.Join(project, "page.ts")
	specFile := filepath.Join(project, "spec.ts")
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('clickSave','function',?,1,1,?)`, pageFile, project)
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('clickCancel','function',?,2,2,?)`, pageFile, project)
	db.IndexDB.Exec(`INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, './page', 'import', ?)`, specFile, project)

	git := func(args ...string) {
		if out, err := exec.Command("git", append([]string{"-C", project}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	git("branch", "-q", "base-branch")
	// A bot-style trim that drops both page-object methods.
	if err := os.WriteFile(pageFile, []byte("export function clickReset() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}

	var got deletionOut
	out := HandleCheckDeletionSafety(map[string]interface{}{"file": "page.ts", "base_ref": "base-branch"}, project)
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal %s: %v", out, err)
	}
	if got.Error != "" {
		t.Fatalf("error: %s", got.Error)
	}
	if len(got.Removed) != 2 || got.Removed[0] != "clickCancel" || got.Removed[1] != "clickSave" {
		t.Fatalf("removed_symbols=%v want [clickCancel clickSave]", got.Removed)
	}
	if refs := got.StillReferenced["clickSave"]; len(refs) != 1 || refs[0] != "spec.ts" {
		t.Fatalf("still_referenced[clickSave]=%v want [spec.ts]", refs)
	}
	if _, unsafe := got.StillReferenced["clickCancel"]; unsafe {
		t.Fatalf("clickCancel has no callers, should be safe: %+v", got)
	}
	if len(got.Safe) != 1 || got.Safe[0] != "clickCancel" {
		t.Fatalf("safe_to_delete=%v want [clickCancel]", got.Safe)
	}
}

func TestHandleCheckDeletionSafetyValidatesArgs(t *testing.T) {
	if out := HandleCheckDeletionSafety(map[string]interface{}{"file": "a.go"}, ""); out != `{"error": "project_path required"}` {
		t.Fatalf("out=%s", out)
	}
	if out := HandleCheckDeletionSafety(map[string]interface{}{}, "/tmp/x"); out != `{"error": "file required"}` {
		t.Fatalf("out=%s", out)
	}
	var got map[string]string
	json.Unmarshal([]byte(HandleCheckDeletionSafety(map[string]interface{}{"file": "notes.md"}, "/tmp/x")), &got)
	if got["error"] == "" {
		t.Fatal("expected unsupported file type error")
	}
	json.Unmarshal([]byte(HandleCheckDeletionSafety(map[string]interface{}{"file": "/etc/hosts"}, "/tmp/x")), &got)
	if got["error"] == "" {
		t.Fatal("expected out-of-project error")
	}
}
