package transcripts

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

const (
	testSelectDailyQuery = `SELECT input, output, cache_read, cache_write FROM host_usage_daily WHERE day = ? AND project_dir = ?`
	repoDir              = "-Users-me-repo"
	otherDir             = "-Users-me-other"
	partialLine          = `{"type":"assistant","timestamp":"2026-10-02T12:10:00Z","requestId":"req_4","message":{"id":"msg_4","usage":{"input_tokens":1,"output_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":4}}}`
)

func copyFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, rel := range []string{repoDir + "/session1.jsonl", otherDir + "/session2.jsonl"} {
		b, err := os.ReadFile(filepath.Join("testdata", "projects", rel))
		require.NoError(t, err)
		require.NoError(t, os.MkdirAll(filepath.Join(root, filepath.Dir(rel)), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(root, rel), b, 0o644))
	}
	return root
}

func localDay(t *testing.T, ts string) string {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339, ts)
	require.NoError(t, err)
	return parsed.Local().Format(dayFormat)
}

func daily(t *testing.T, day, project string) Usage {
	t.Helper()
	u := Usage{Day: day}
	require.NoError(t, db.DB.QueryRow(testSelectDailyQuery, day, project).Scan(&u.Input, &u.Output, &u.CacheRead, &u.CacheWrite))
	return u
}

func TestIngestOnceAggregatesUsage(t *testing.T) {
	dbtest.Init(t)
	root := copyFixture(t)
	res, err := IngestOnce(root)
	require.NoError(t, err)
	assert.Equal(t, 2, res.Files)
	day1, day2 := localDay(t, "2026-10-01T12:00:05Z"), localDay(t, "2026-10-02T12:00:00Z")
	// msg_1 appears on two content-block lines and counts once.
	assert.Equal(t, Usage{Day: day1, Input: 100, Output: 20, CacheRead: 1000, CacheWrite: 50}, daily(t, day1, repoDir))
	assert.Equal(t, Usage{Day: day2, Input: 10, Output: 5}, daily(t, day2, repoDir))
	assert.Equal(t, Usage{Day: day1, Input: 7, Output: 3, CacheRead: 200}, daily(t, day1, otherDir))

	// A second pass over unchanged files adds nothing.
	res, err = IngestOnce(root)
	require.NoError(t, err)
	assert.Zero(t, res.Files)
	assert.Equal(t, int64(100), daily(t, day1, repoDir).Input)

	// A partial last line waits until it is terminated.
	path := filepath.Join(root, repoDir, "session1.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	require.NoError(t, err)
	_, err = f.WriteString(partialLine)
	require.NoError(t, err)
	_, err = IngestOnce(root)
	require.NoError(t, err)
	assert.Equal(t, int64(10), daily(t, day2, repoDir).Input)
	_, err = f.WriteString("\n")
	require.NoError(t, err)
	require.NoError(t, f.Close())
	_, err = IngestOnce(root)
	require.NoError(t, err)
	assert.Equal(t, Usage{Day: day2, Input: 11, Output: 7, CacheRead: 3, CacheWrite: 4}, daily(t, day2, repoDir))

	series, err := DailySeries(36500)
	require.NoError(t, err)
	require.Len(t, series, 2)
	assert.Equal(t, Usage{Day: day1, Input: 107, Output: 23, CacheRead: 1200, CacheWrite: 50}, series[0])
}

func TestRunOnceRespectsSetting(t *testing.T) {
	home := dbtest.Init(t)
	assert.False(t, Enabled())
	root := filepath.Join(home, ".claude", "projects")
	require.NoError(t, os.MkdirAll(filepath.Dir(root), 0o755))
	require.NoError(t, os.CopyFS(root, os.DirFS(filepath.Join("testdata", "projects"))))
	RunOnce()
	series, err := DailySeries(36500)
	require.NoError(t, err)
	assert.Empty(t, series)
	require.NoError(t, db.SetSetting(SettingKey, "true"))
	RunOnce()
	series, err = DailySeries(36500)
	require.NoError(t, err)
	assert.Len(t, series, 2)
}
