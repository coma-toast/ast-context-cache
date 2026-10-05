package context

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestMarkReturnedIsVisibleImmediately(t *testing.T) {
	dbtest.Init(t)
	sid := t.Name()
	MarkReturned(sid, ReturnedSymbol{File: "/p/a.go", Name: "Foo", StartLine: 3, Mode: "skeleton", Tokens: 10})
	_, ok := ReturnedKeys(sid)["/p/a.go|Foo|3"]
	assert.True(t, ok, "no wait for the write buffer")
	keys := ReturnedKeys(sid)
	keys["/p/b.go|Bar|1"] = struct{}{}
	_, leaked := ReturnedKeys(sid)["/p/b.go|Bar|1"]
	assert.False(t, leaked, "ReturnedKeys hands out a copy")
	assert.Empty(t, ReturnedKeys(""))
}

// A restart loses the in-memory sets; the next call rehydrates from the sessions table.
func TestReturnedKeysHydrateAfterRestart(t *testing.T) {
	dbtest.Init(t)
	sid := t.Name()
	MarkReturned(sid, ReturnedSymbol{File: "/p/a.go", Name: "Foo", StartLine: 3, Mode: "full", Tokens: 10})
	SeedReturned(sid, []ReturnedSymbol{{File: "/p/b.go", Name: "Bar", StartLine: 9}})
	db.FlushWriteBuffers()
	sessions.Clear()
	keys := ReturnedKeys(sid)
	assert.Contains(t, keys, "/p/a.go|Foo|3")
	assert.Contains(t, keys, "/p/b.go|Bar|9")
	var mode string
	require.NoError(t, db.DB.QueryRow(`SELECT mode FROM sessions WHERE session_id = ? AND symbol_name = 'Bar'`, sid).Scan(&mode))
	assert.Equal(t, "seed", mode)
}

func TestEvictIdleSessions(t *testing.T) {
	dbtest.Init(t)
	sid := t.Name()
	MarkReturned(sid, ReturnedSymbol{File: "/p/a.go", Name: "Foo", StartLine: 3})
	assert.Zero(t, evictIdleSessions(time.Now().Add(-time.Hour)), "recently used sets stay")
	assert.GreaterOrEqual(t, evictIdleSessions(time.Now().Add(time.Second)), 1)
	_, loaded := sessions.Load(sid)
	assert.False(t, loaded)
	db.FlushWriteBuffers()
	assert.Contains(t, ReturnedKeys(sid), "/p/a.go|Foo|3", "an evicted session rehydrates")
}
