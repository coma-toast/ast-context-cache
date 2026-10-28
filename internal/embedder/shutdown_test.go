package embedder

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stoppable struct{ stops int }

func (s *stoppable) Embed([]string) ([][]float32, error)   { return nil, nil }
func (s *stoppable) EmbedSingle(string) ([]float32, error) { return nil, nil }
func (s *stoppable) Stop()                                 { s.stops++ }

func TestLazyEmbedderStopIsFinalAndIdempotent(t *testing.T) {
	le := NewLazy(t.TempDir())
	le.Stop()
	le.Stop()
	_, err := le.EmbedSingle("x")
	require.Error(t, err)
	assert.True(t, errors.Is(err, errEmbedderStopped), "a stopped embedder must not lazy-load a new session")
	_, err = le.Embed([]string{"x"})
	assert.True(t, errors.Is(err, errEmbedderStopped))
	assert.False(t, le.IsLoaded())
}

func TestShutdownStopsPrimaryAuxAndRuntime(t *testing.T) {
	primary, aux := &stoppable{}, &stoppable{}
	destroyed := 0
	origDestroy := destroyRuntime
	destroyRuntime = func() { destroyed++ }
	runtimeMu.Lock()
	origRaw, origTracked, origLoaded := runtimeRaw, runtimeTracked, runtimeLoaded
	runtimeRaw, runtimeTracked, runtimeLoaded = primary, TrackHealth(primary), func() bool { return true }
	runtimeMu.Unlock()
	auxMu.Lock()
	origAux := auxRaw
	auxRaw = aux
	auxMu.Unlock()
	t.Cleanup(func() {
		destroyRuntime = origDestroy
		runtimeMu.Lock()
		runtimeRaw, runtimeTracked, runtimeLoaded = origRaw, origTracked, origLoaded
		runtimeMu.Unlock()
		auxMu.Lock()
		auxRaw = origAux
		auxMu.Unlock()
	})

	Shutdown()
	assert.Equal(t, 1, primary.stops, "the tracked wrapper is unwrapped and its embedder stopped")
	assert.Equal(t, 1, aux.stops)
	assert.Equal(t, 1, destroyed, "the ONNX Runtime environment is torn down last")
	assert.Nil(t, Tracked())
	assert.Nil(t, Raw())
	assert.Nil(t, RawAux())
	assert.False(t, IsLoaded())
}
