package db

import (
	"database/sql"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	testCreateT1Query           = `CREATE TABLE t1 (id INTEGER)`
	testCreateT2Query           = `CREATE TABLE t2 (id INTEGER)`
	testCreateT3Query           = `CREATE TABLE t3 (id INTEGER)`
	testBadQuery                = `INSERT INTO no_such_table VALUES (1)`
	testCountTableQuery         = `SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?`
	testInsertNoteQuery         = `INSERT INTO context_notes (ref, session_id, content, content_hash, last_accessed_at) VALUES (?, 's', 'c', 'h', ?)`
	testSelectNoteAccessedQuery = `SELECT last_accessed_at FROM context_notes WHERE ref = ?`
	testInsertNoteAccessQuery   = `INSERT INTO context_note_access (ref, tool_name, virtual_tokens, accessed_at) VALUES (?, 'fetch_context', 1, ?)`
	testSelectNoteAccessQuery   = `SELECT accessed_at FROM context_note_access WHERE ref = ?`
	testInsertSessionStatsQuery = `INSERT INTO context_session_stats (session_id, last_store_at, last_access_at) VALUES (?, ?, ?)`
	testSelectSessionStatsQuery = `SELECT last_store_at, last_access_at FROM context_session_stats WHERE session_id = ?`
	testInsertOrphanDocQuery    = `INSERT INTO doc_content (source_id, title, content) VALUES (999, 'orphan', 'x')`
	testCountDocContentQuery    = `SELECT COUNT(*) FROM doc_content`
	testSelectEstimateMethod    = `SELECT estimate_method FROM queries WHERE tool_name = ?`
)

func stepTempDB(t *testing.T) *sql.DB {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	conn, err := openStepPool(filepath.Join(t.TempDir(), "steps.db"))
	require.NoError(t, err)
	t.Cleanup(func() { conn.Close() })
	return conn
}

func tableExists(t *testing.T, conn *sql.DB, name string) bool {
	t.Helper()
	var n int
	require.NoError(t, conn.QueryRow(testCountTableQuery, name).Scan(&n))
	return n == 1
}

func TestRunStepsAppliesPendingStepsOnce(t *testing.T) {
	conn := stepTempDB(t)
	calls := 0
	counted := func(q string) func(tx *sql.Tx) error {
		return func(tx *sql.Tx) error {
			calls++
			return execSteps(q)(tx)
		}
	}
	steps := []schemaStep{{1, "one", counted(testCreateT1Query)}, {2, "two", counted(testCreateT2Query)}}
	require.NoError(t, runSteps(conn, "test", steps))
	require.NoError(t, runSteps(conn, "test", steps))
	assert.Equal(t, 2, calls)
	steps = append(steps, schemaStep{3, "three", counted(testCreateT3Query)})
	require.NoError(t, runSteps(conn, "test", steps))
	assert.Equal(t, 3, calls)
	v, err := userVersion(conn)
	require.NoError(t, err)
	assert.Equal(t, 3, v)
	assert.True(t, tableExists(t, conn, "t3"))
}

func TestRunStepsRollsBackFailedStep(t *testing.T) {
	conn := stepTempDB(t)
	steps := []schemaStep{
		{1, "one", execSteps(testCreateT1Query)},
		{2, "broken", execSteps(testCreateT2Query, testBadQuery)},
		{3, "three", execSteps(testCreateT3Query)},
	}
	err := runSteps(conn, "test", steps)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "schema step failed")
	assert.Equal(t, "broken", errs.FieldsOf(err)["step"])
	assert.Equal(t, "test", errs.FieldsOf(err)["db"])
	v, err := userVersion(conn)
	require.NoError(t, err)
	assert.Equal(t, 1, v)
	assert.True(t, tableExists(t, conn, "t1"))
	assert.False(t, tableExists(t, conn, "t2"), "failed step's work must roll back")
	assert.False(t, tableExists(t, conn, "t3"), "no step runs after a failure")
}

func TestStepListsAreContiguous(t *testing.T) {
	for name, steps := range map[string][]schemaStep{"index": indexSteps, "context": contextSteps, "usage": usageSteps} {
		for i, s := range steps {
			assert.Equal(t, i+1, s.version, "%s step %q", name, s.name)
			assert.NotEmpty(t, s.name)
			assert.NotNil(t, s.run)
		}
	}
}

// DB-4: schema changes stay additive so a 4.x binary still runs against a 5.x data dir.
func TestStepSQLIsAdditive(t *testing.T) {
	forbidden := regexp.MustCompile(`(?i)\bDROP\b|\bRENAME\b|\bALTER\s+COLUMN\b`)
	f, err := parser.ParseFile(token.NewFileSet(), "migrate_steps.go", nil, 0)
	require.NoError(t, err)
	found := 0
	for _, decl := range f.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.CONST {
			continue
		}
		ast.Inspect(gd, func(n ast.Node) bool {
			lit, ok := n.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				return true
			}
			sqlText, err := strconv.Unquote(lit.Value)
			require.NoError(t, err)
			found++
			assert.False(t, forbidden.MatchString(sqlText), "non-additive schema step SQL: %s", sqlText)
			return true
		})
	}
	assert.Positive(t, found, "no step SQL constants found in migrate_steps.go")
}

// seedV4DataDir writes 4.x-shaped context.db and usage.db (user_version 0) under a fresh HOME.
func seedV4DataDir(t *testing.T, seed func(ctxConn, useConn *sql.DB)) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	require.NoError(t, os.MkdirAll(cacheDir(), 0o755))
	ctxConn, err := sql.Open("sqlite3", contextDBPath())
	require.NoError(t, err)
	defer ctxConn.Close()
	useConn, err := sql.Open("sqlite3", usageDBPath())
	require.NoError(t, err)
	defer useConn.Close()
	initContextSchema(ctxConn)
	initUsageSchema(useConn)
	seed(ctxConn, useConn)
}

func plainUserVersion(t *testing.T, path string) int {
	t.Helper()
	conn, err := sql.Open("sqlite3", path)
	require.NoError(t, err)
	defer conn.Close()
	v, err := userVersion(conn)
	require.NoError(t, err)
	return v
}

func TestInitMigratesV4DataDir(t *testing.T) {
	seedV4DataDir(t, func(ctxConn, useConn *sql.DB) {
		_, err := ctxConn.Exec(testInsertNoteQuery, "ctx_rfc", "2026-10-09T15:04:05-05:00")
		require.NoError(t, err)
		_, err = ctxConn.Exec(testInsertNoteQuery, "ctx_sql", "2026-10-09 01:02:03")
		require.NoError(t, err)
		_, err = ctxConn.Exec(testInsertOrphanDocQuery)
		require.NoError(t, err)
		_, err = useConn.Exec(testInsertNoteAccessQuery, "ctx_rfc", "2026-10-09T15:04:05Z")
		require.NoError(t, err)
		_, err = useConn.Exec(testInsertSessionStatsQuery, "s1", "2026-10-09T15:04:05Z", "2026-10-09T16:04:05.5+01:00")
		require.NoError(t, err)
	})
	t.Cleanup(Close)
	require.NoError(t, Init())

	// BF-5 normalization (context#1, usage#1).
	var got string
	require.NoError(t, ContextDB.QueryRow(testSelectNoteAccessedQuery, "ctx_rfc").Scan(&got))
	assert.Equal(t, "2026-10-09 20:04:05", got)
	require.NoError(t, ContextDB.QueryRow(testSelectNoteAccessedQuery, "ctx_sql").Scan(&got))
	assert.Equal(t, "2026-10-09 01:02:03", got)
	require.NoError(t, DB.QueryRow(testSelectNoteAccessQuery, "ctx_rfc").Scan(&got))
	assert.Equal(t, "2026-10-09 15:04:05", got)
	var store, access string
	require.NoError(t, DB.QueryRow(testSelectSessionStatsQuery, "s1").Scan(&store, &access))
	assert.Equal(t, "2026-10-09 15:04:05", store)
	assert.Equal(t, "2026-10-09 15:04:05", access)
	assert.Equal(t, len(contextSteps), plainUserVersion(t, contextDBPath()))
	assert.Equal(t, len(usageSteps), plainUserVersion(t, usageDBPath()))

	// DB-2: the orphaned row is counted and kept.
	assert.Equal(t, 1, ForeignKeyViolations())
	var docs int
	require.NoError(t, ContextDB.QueryRow(testCountDocContentQuery).Scan(&docs))
	assert.Equal(t, 1, docs)

	// DB-5: the 4.x databases were copied aside before the steps ran.
	snap, err := sql.Open("sqlite3", filepath.Join(pre50SnapshotDir(), "context.db"))
	require.NoError(t, err)
	defer snap.Close()
	require.NoError(t, snap.QueryRow(testSelectNoteAccessedQuery, "ctx_rfc").Scan(&got))
	assert.Equal(t, "2026-10-09T15:04:05-05:00", got)
	assert.FileExists(t, filepath.Join(pre50SnapshotDir(), "usage.db"))
	assert.NoFileExists(t, filepath.Join(pre50SnapshotDir(), "context.db.tmp"))

	// usage#2: new query-log rows record the estimator.
	LogQuery("estimate_method_tool", nil, QueryLogMetrics{}, "/p", "")
	FlushWriteBuffers()
	require.NoError(t, DB.QueryRow(testSelectEstimateMethod, "estimate_method_tool").Scan(&got))
	assert.Equal(t, EstimateMethod(), got)
}

func TestInitFreshDataDirTakesNoSnapshot(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	t.Cleanup(Close)
	require.NoError(t, Init())
	assert.NoDirExists(t, filepath.Join(cacheDir(), snapshotsDirName))
	assert.Equal(t, len(contextSteps), plainUserVersion(t, contextDBPath()))
	assert.Equal(t, len(usageSteps), plainUserVersion(t, usageDBPath()))
	assert.Equal(t, 0, ForeignKeyViolations())
	// A second start neither snapshots nor re-runs steps.
	Close()
	require.NoError(t, Init())
	assert.NoDirExists(t, filepath.Join(cacheDir(), snapshotsDirName))
}

func TestInitFailsWithoutRunningStepsWhenSnapshotFails(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	seedV4DataDir(t, func(ctxConn, useConn *sql.DB) {
		_, err := ctxConn.Exec(testInsertNoteQuery, "ctx_rfc", "2026-10-09T15:04:05Z")
		require.NoError(t, err)
	})
	snapRoot := filepath.Join(cacheDir(), snapshotsDirName)
	require.NoError(t, os.MkdirAll(snapRoot, 0o500))
	t.Cleanup(func() { os.Chmod(snapRoot, 0o755) })
	t.Cleanup(Close)
	err := Init()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "pre-5.0 snapshot")
	assert.Equal(t, 0, plainUserVersion(t, contextDBPath()))
	assert.Equal(t, 0, plainUserVersion(t, usageDBPath()))
}
