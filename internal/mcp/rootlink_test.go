package mcp

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

// JD-254: index_files with a project_path that is a symlink to a directory indexed
// 0 files and reported success. The project is keyed under the link's target, and
// index_status and search answer for either spelling.
func TestIndexFilesThroughSymlinkedProjectRoot(t *testing.T) {
	dbtest.Init(t)
	isolateIndexJobs(t)
	base := t.TempDir()
	real, link := filepath.Join(base, "real"), filepath.Join(base, "link")
	if err := os.MkdirAll(filepath.Join(real, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package demo\n\nfunc LinkedRootGreeting() string { return \"hi\" }\n"
	if err := os.WriteFile(filepath.Join(real, "sub", "demo.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	out, isErr := callTool(t, "index_files", map[string]interface{}{"path": link, "project_path": link})
	if isErr || out["status"] != indexJobCompleted || out["files_done"] != float64(1) {
		t.Fatalf("want a completed job over 1 file, got isErr=%v %v", isErr, out)
	}
	if out["project_path"] != real || out["path"] != real || out["warning"] != nil {
		t.Fatalf("job should be keyed under %s with no warning: %v", real, out)
	}
	for _, tc := range []struct{ name, project string }{{"link", link}, {"real", real}} {
		t.Run(tc.name, func(t *testing.T) {
			st, isErr := callTool(t, "index_status", map[string]interface{}{"project_path": tc.project})
			if isErr || st["own_files"] != float64(1) {
				t.Fatalf("index_status(%s) own_files=%v, want 1: %v", tc.project, st["own_files"], st)
			}
			res, isErr := callTool(t, "get_context_capsule", map[string]interface{}{"query": "LinkedRootGreeting", "project_path": tc.project})
			if hits, _ := res["results"].([]interface{}); isErr || len(hits) == 0 {
				t.Fatalf("get_context_capsule(%s) found nothing: %v", tc.project, res)
			}
		})
	}
	// A single file named through the link is respelled under the real project.
	file, isErr := callTool(t, "index_files", map[string]interface{}{"path": filepath.Join(link, "sub", "demo.go"), "project_path": link})
	if isErr || file["project_path"] != real || file["indexed"] == float64(0) {
		t.Fatalf("file via link: isErr=%v %v", isErr, file)
	}
}

// A walk that finds no code files completes with a warning instead of a silent
// success.
func TestIndexFilesWarnsWhenNoCodeFiles(t *testing.T) {
	dbtest.Init(t)
	isolateIndexJobs(t)
	project := t.TempDir()
	if err := os.WriteFile(filepath.Join(project, "notes.bin"), []byte{0, 1, 2}, 0o644); err != nil {
		t.Fatal(err)
	}
	out, isErr := callTool(t, "index_files", map[string]interface{}{"path": project, "project_path": project})
	if isErr || out["status"] != indexJobCompleted || out["warning"] != noCodeFilesWarning {
		t.Fatalf("want completed with warning, got isErr=%v %v", isErr, out)
	}
}
