package purge

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// mkSpace lays out a wtg space: go.work plus a worktree-style checkout (.git file)
// per name.
func mkSpace(t *testing.T, dir string, checkouts ...string) []string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.work"), []byte("go 1.23\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, c := range checkouts {
		p := filepath.Join(dir, c)
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(p, ".git"), []byte("gitdir: x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		out = append(out, p)
	}
	return out
}

// seedFile indexes one file of projectPath (symbol, edge, vector, indexed file).
func seedFile(t *testing.T, projectPath, file string) {
	t.Helper()
	for _, q := range []struct {
		sql  string
		args []interface{}
	}{
		{`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('X','function',?,1,1,?)`, []interface{}{file, projectPath}},
		{`INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, 'fmt', 'import', ?)`, []interface{}{file, projectPath}},
		{`INSERT INTO indexed_files (file, project_path, indexed_at) VALUES (?, ?, datetime('now'))`, []interface{}{file, projectPath}},
		{`INSERT INTO vectors (content_hash, vector, doc_type, source_file, name, kind, project_path) VALUES (?, ?, 'code', ?, 'X', 'function', ?)`, []interface{}{"h-" + file, []byte{1, 2, 3, 4}, file, projectPath}},
	} {
		if _, err := db.IndexDB.Exec(q.sql, q.args...); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSpaceRootsDryRunListsCoveredRootsOnly(t *testing.T) {
	home := dbtest.Init(t)
	covered := filepath.Join(home, "spaces", "covered")
	partial := filepath.Join(home, "spaces", "partial")
	plain := filepath.Join(home, "notes")
	coveredKids := mkSpace(t, covered, "api", "web")
	partialKids := mkSpace(t, partial, "api", "web")
	if err := os.MkdirAll(plain, 0o755); err != nil {
		t.Fatal(err)
	}
	// Each space root was indexed whole: rows for files inside its checkouts plus
	// a loose file at the root.
	for _, root := range []string{covered, partial} {
		seedFile(t, root, filepath.Join(root, "api", "a.go"))
		seedFile(t, root, filepath.Join(root, "plan.md"))
	}
	for _, k := range coveredKids {
		seedFile(t, k, filepath.Join(k, "a.go"))
	}
	seedFile(t, partialKids[0], filepath.Join(partialKids[0], "a.go"))
	seedFile(t, plain, filepath.Join(plain, "todo.md"))
	// The root keeps its notes; pruning only drops index rows.
	db.ContextDB.Exec(`INSERT INTO context_notes (ref, session_id, project_path, label, content, content_hash) VALUES ('ctx_root', 'sess', ?, 'l', 'c', 'h')`, covered)

	roots := SpaceRoots(false)
	if len(roots) != 2 {
		t.Fatalf("roots=%+v want the two space roots, not the plain folder", roots)
	}
	byPath := map[string]SpaceRoot{}
	for _, r := range roots {
		byPath[r.Path] = r
	}
	if r := byPath[covered]; !r.Covered || r.Pruned || r.Files != 2 || len(r.Checkouts) != 2 {
		t.Fatalf("covered root=%+v want covered, unpruned, 2 files, 2 checkouts", r)
	}
	if r := byPath[partial]; r.Covered || len(r.Unindexed) != 1 || r.Unindexed[0] != partialKids[1] {
		t.Fatalf("partial root=%+v want not covered with %s unindexed", r, partialKids[1])
	}
	if n := symbolCount(t, covered); n != 2 {
		t.Fatalf("dry run deleted rows: covered root has %d symbols, want 2", n)
	}

	roots = SpaceRoots(true)
	for _, r := range roots {
		if r.Path == covered && !r.Pruned {
			t.Fatalf("covered root not pruned: %+v", r)
		}
		if r.Path == partial && r.Pruned {
			t.Fatalf("partial root pruned: %+v", r)
		}
	}
	for _, table := range []string{"symbols", "edges", "indexed_files", "vectors"} {
		var n int
		db.IndexDB.QueryRow(`SELECT COUNT(*) FROM `+table+` WHERE project_path = ?`, covered).Scan(&n)
		if n != 0 {
			t.Fatalf("%s still has %d rows for the pruned root", table, n)
		}
	}
	if n := symbolCount(t, partial); n != 2 {
		t.Fatalf("partial root has %d symbols, want 2 (untouched)", n)
	}
	for _, k := range coveredKids {
		if n := symbolCount(t, k); n != 1 {
			t.Fatalf("checkout %s has %d symbols, want 1", k, n)
		}
	}
	if links, _ := projectlinks.Links(covered); len(links) != 2 {
		t.Fatalf("links=%v want both checkouts linked so the root still searches them", links)
	}
	var notes int
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_notes WHERE project_path = ?`, covered).Scan(&notes)
	if notes != 1 {
		t.Fatalf("root notes=%d want 1 kept", notes)
	}
}
