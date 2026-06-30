package watcher

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestGetStatusIncludesIndexedProjectWithoutWatcher(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatalf("db init: %v", err)
	}
	dir := t.TempDir()
	projectPath := NormalizeProjectPath(dir)
	file := filepath.Join(dir, "sample.go")
	if err := os.WriteFile(file, []byte("package sample\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES (?, ?, ?, ?, ?, ?)`,
		"Foo", "function", file, 1, 1, projectPath); err != nil {
		t.Fatalf("insert symbol: %v", err)
	}
	DeleteWatcher(projectPath)

	st := GetStatus()
	watchers, ok := st["watchers"].([]map[string]interface{})
	if !ok {
		t.Fatalf("watchers type: %T", st["watchers"])
	}
	var found bool
	for _, w := range watchers {
		pp, _ := w["project_path"].(string)
		if pp != projectPath {
			continue
		}
		found = true
		if active, _ := w["active"].(bool); active {
			t.Fatal("expected inactive watcher row for indexed-only project")
		}
	}
	if !found {
		t.Fatalf("indexed project %q missing from watcher status", projectPath)
	}
}
