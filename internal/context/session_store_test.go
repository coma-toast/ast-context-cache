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

// MO-4: the session keeps the richest mode each symbol was delivered in, across a restart.
func TestReturnedModesHydrateRichest(t *testing.T) {
	dbtest.Init(t)
	sid := t.Name()
	MarkReturned(sid, ReturnedSymbol{File: "/p/a.go", Name: "Foo", StartLine: 3, Mode: "full"})
	MarkReturned(sid, ReturnedSymbol{File: "/p/a.go", Name: "Foo", StartLine: 3, Mode: "skeleton"})
	MarkReturned(sid, ReturnedSymbol{File: "/p/b.go", Name: "Bar", StartLine: 1, Mode: "summary"})
	SeedReturned(sid, []ReturnedSymbol{{File: "/p/c.go", Name: "Baz", StartLine: 2}})
	want := map[string]string{"/p/a.go|Foo|3": "full", "/p/b.go|Bar|1": "summary", "/p/c.go|Baz|2": "seed"}
	assert.Equal(t, want, ReturnedModes(sid))
	db.FlushWriteBuffers()
	sessions.Clear()
	assert.Equal(t, want, ReturnedModes(sid), "hydrated from the sessions table")
	modes := ReturnedModes(sid)
	assert.True(t, DedupCovers(modes, "/p/a.go|Foo|3", "full"))
	assert.False(t, DedupCovers(modes, "/p/b.go|Bar|1", "skeleton"), "a summary does not cover a skeleton")
	assert.True(t, DedupCovers(modes, "/p/c.go|Baz|2", "full"), "a seed covers any mode")
	assert.False(t, DedupCovers(modes, "/p/d.go|Qux|1", "locations"))
	assert.Empty(t, ReturnedModes(""))
}
