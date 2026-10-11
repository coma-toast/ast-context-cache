package db

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// applyPragmas used to Exec on the pool, so the settings reached whichever single
// connection served the call. The driver's ConnectHook now applies them to each one.
func TestPoolAppliesPragmasOnEveryConnection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	pool, err := openPool(filepath.Join(t.TempDir(), "pragmas.db"))
	require.NoError(t, err)
	t.Cleanup(func() { pool.Close() })
	ctx := context.Background()
	conns := make([]*sql.Conn, 4)
	for i := range conns {
		c, err := pool.Conn(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { c.Close() })
		conns[i] = c
	}
	pragmas := map[string]string{
		"cache_size": "-32000", "busy_timeout": "15000", "synchronous": "1",
		"wal_autocheckpoint": "200", "foreign_keys": "1", "journal_mode": "wal",
	}
	for i, c := range conns {
		for name, want := range pragmas {
			var got string
			require.NoError(t, c.QueryRowContext(ctx, "PRAGMA "+name).Scan(&got))
			assert.Equal(t, want, got, "conn %d pragma %s", i, name)
		}
	}
}
