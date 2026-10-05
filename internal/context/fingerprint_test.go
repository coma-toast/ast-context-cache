package context

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func TestSymbolFingerprint(t *testing.T) {
	t.Parallel()
	file := filepath.Join(t.TempDir(), "a.go")
	write := func(src string) {
		t.Helper()
		require.NoError(t, os.WriteFile(file, []byte(src), 0o644))
	}
	write("package a\n\nfunc Foo() int {\n\treturn 1\n}\n")
	fp, err := SymbolFingerprint(file, 3, 5)
	require.NoError(t, err)
	assert.Len(t, fp, 32, "16 bytes of sha256, hex")
	again, err := SymbolFingerprint(file, 3, 5)
	require.NoError(t, err)
	assert.Equal(t, fp, again, "stable")
	write("// moved\npackage a\n\nfunc Foo() int {\n\treturn 1\n}\n")
	moved, err := SymbolFingerprint(file, 4, 6)
	require.NoError(t, err)
	assert.Equal(t, fp, moved, "same text at new lines")
	write("package a\n\nfunc Foo() int {\n\treturn 2\n}\n")
	edited, err := SymbolFingerprint(file, 3, 5)
	require.NoError(t, err)
	assert.NotEqual(t, fp, edited, "edit changes it")
	_, err = SymbolFingerprint(file, 9, 12)
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
	_, err = SymbolFingerprint(filepath.Join(t.TempDir(), "gone.go"), 1, 1)
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
}

func TestLookupSymbol(t *testing.T) {
	dbtest.Init(t)
	// Init's background FTS rebuild holds the index write lock; seeding symbols must wait for it.
	db.WaitInitFTSRebuildForTest()
	project := "/proj"
	require.NoError(t, db.IndexWrite(func(tx *sql.Tx) error {
		for _, s := range []struct {
			name, fqn, kind string
			start, end      int
		}{
			{"Run", "svc.go.Server.Run", "method", 10, 20},
			{"Run", "svc.go.Client.Run", "method", 30, 40},
			{"helper", "svc.go.helper", "function", 50, 55},
		} {
			if _, err := tx.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?)`,
				s.name, s.kind, "/proj/internal/svc.go", s.start, s.end, s.fqn, project); err != nil {
				return err
			}
		}
		return nil
	}))
	r, err := LookupSymbol(project, "internal/svc.go", "svc.go.Client.Run", "Run")
	require.NoError(t, err)
	assert.Equal(t, SymbolRow{
		File: "/proj/internal/svc.go", FileRel: "internal/svc.go", ProjectPath: project, Name: "Run",
		FQN: "svc.go.Client.Run", Kind: "method", StartLine: 30, EndLine: 40,
	}, *r)
	r, err = LookupSymbol(project, "/proj/internal/svc.go", "", "helper")
	require.NoError(t, err)
	assert.Equal(t, 50, r.StartLine, "absolute path and name-only lookup")
	r, err = LookupSymbol(project, "internal/svc.go", "svc.go.Renamed.Run", "Run")
	require.NoError(t, err)
	assert.Equal(t, 10, r.StartLine, "unknown fqn falls back to the first symbol with the name")
	_, err = LookupSymbol(project, "internal/svc.go", "svc.go.gone", "gone")
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
	_, err = LookupSymbol(project, "internal/other.go", "", "Run")
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
	_, err = LookupSymbol(project, "", "", "Run")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
}

func TestDeleteSessionKeys(t *testing.T) {
	dbtest.Init(t)
	MarkReturned("child-a", ReturnedSymbol{File: "/p/a.go", Name: "A", StartLine: 1})
	MarkReturned("child-b", ReturnedSymbol{File: "/p/b.go", Name: "B", StartLine: 2})
	MarkReturned("keep", ReturnedSymbol{File: "/p/c.go", Name: "C", StartLine: 3})
	require.NoError(t, DeleteSessionKeys("child-a", "child-b", ""))
	assert.Empty(t, ReturnedKeys("child-a"), "in-memory set and buffered rows both gone")
	assert.Empty(t, ReturnedKeys("child-b"))
	assert.Contains(t, ReturnedKeys("keep"), "/p/c.go|C|3")
	db.FlushWriteBuffers()
	var n int
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM sessions WHERE session_id IN ('child-a', 'child-b')`).Scan(&n))
	assert.Zero(t, n)
	require.NoError(t, db.DB.QueryRow(`SELECT COUNT(*) FROM sessions WHERE session_id = 'keep'`).Scan(&n))
	assert.Equal(t, 1, n)
	require.NoError(t, DeleteSessionKeys())
}
