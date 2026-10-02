package db

import (
	"database/sql"
	"path/filepath"
	"testing"
)

// A monolithic DB written before indexed_files.parser_version existed must still
// migrate: the copy goes by column name, and missing columns take their default.
func TestCopyTablesFromAttachToleratesAddedColumns(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "usage.db")
	src, err := sql.Open("sqlite3", srcPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := src.Exec(`CREATE TABLE indexed_files (file TEXT NOT NULL, project_path TEXT NOT NULL, indexed_at DATETIME NOT NULL, PRIMARY KEY (file, project_path));
		INSERT INTO indexed_files VALUES ('/p/a.py', '/p', '2026-01-01T00:00:00Z');`); err != nil {
		t.Fatal(err)
	}
	src.Close()

	dest, err := openPool(filepath.Join(dir, "index.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer dest.Close()
	initIndexSchema(dest)
	if err := copyTablesFromAttach(dest, srcPath, []string{"indexed_files"}); err != nil {
		t.Fatal(err)
	}
	var file string
	var version int
	if err := dest.QueryRow(`SELECT file, parser_version FROM indexed_files`).Scan(&file, &version); err != nil {
		t.Fatal(err)
	}
	if file != "/p/a.py" || version != 0 {
		t.Fatalf("file=%q parser_version=%d", file, version)
	}
}
