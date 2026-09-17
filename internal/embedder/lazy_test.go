package embedder

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

// testModelDir resolves the repo's real (gitignored, locally-downloaded) ONNX
// model directory relative to this test file, skipping if it isn't present —
// this test needs a real *Embedder, not a stub, to exercise the actual
// lazy-load/close lifecycle.
func testModelDir(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot resolve test file path")
	}
	dir := filepath.Join(filepath.Dir(thisFile), "..", "..", "model")
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err != nil {
		t.Skipf("real model not available at %s: %v", dir, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("real tokenizer not available at %s: %v", dir, err)
	}
	return dir
}

// getLocked used to release le.mu right after fetching le.inner, before the
// actual Embed/EmbedSingle call ran — so idleLoop's ticker (or a manual Close)
// could Close() the same *Embedder instance while a caller was still mid-call
// on it. Embed/EmbedSingle now hold le.mu for the whole call, so a concurrent
// Close() can never overlap with an in-flight embed.
func TestLazyEmbedderCloseDuringEmbedDoesNotRace(t *testing.T) {
	modelDir := testModelDir(t)
	le := NewLazy(modelDir)
	defer le.Stop()

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := le.EmbedSingle("race test text"); err != nil {
					t.Errorf("EmbedSingle: %v", err)
					return
				}
			}
		}()
	}

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		le.Close()
	}
	close(stop)
	wg.Wait()
}
