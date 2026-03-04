package embedder

import (
	"log"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

type LazyEmbedder struct {
	mu       sync.Mutex
	modelDir string
	inner    *Embedder
	lastUsed time.Time
	stopIdle chan struct{}
}

func NewLazy(modelDir string) *LazyEmbedder {
	le := &LazyEmbedder{modelDir: modelDir, stopIdle: make(chan struct{})}
	go le.idleLoop()
	return le
}

func (le *LazyEmbedder) get() (*Embedder, error) {
	le.mu.Lock()
	defer le.mu.Unlock()
	le.lastUsed = time.Now()
	if le.inner != nil {
		return le.inner, nil
	}
	log.Printf("Lazy-loading embedder from %s...", le.modelDir)
	e, err := New(le.modelDir)
	if err != nil {
		return nil, err
	}
	le.inner = e
	log.Printf("Embedder loaded: %s (%d dims)", ModelName, Dimensions)
	return e, nil
}

func (le *LazyEmbedder) Embed(texts []string) ([][]float32, error) {
	e, err := le.get()
	if err != nil {
		return nil, err
	}
	return e.Embed(texts)
}

func (le *LazyEmbedder) EmbedSingle(text string) ([]float32, error) {
	e, err := le.get()
	if err != nil {
		return nil, err
	}
	return e.EmbedSingle(text)
}

func (le *LazyEmbedder) IsLoaded() bool {
	le.mu.Lock()
	defer le.mu.Unlock()
	return le.inner != nil
}

func (le *LazyEmbedder) Close() {
	le.mu.Lock()
	defer le.mu.Unlock()
	if le.inner != nil {
		le.inner.Close()
		le.inner = nil
		log.Println("Embedder unloaded (idle timeout)")
	}
}

func (le *LazyEmbedder) Stop() {
	close(le.stopIdle)
	le.Close()
}

func (le *LazyEmbedder) idleTimeout() time.Duration {
	val := db.GetSetting("idle_unload_minutes", "1")
	mins, err := strconv.Atoi(val)
	if err != nil || mins < 0 {
		mins = 1
	}
	if mins == 0 {
		return 0
	}
	return time.Duration(mins) * time.Minute
}

func (le *LazyEmbedder) idleLoop() {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			timeout := le.idleTimeout()
			if timeout == 0 {
				continue
			}
			le.mu.Lock()
			if le.inner != nil && time.Since(le.lastUsed) > timeout {
				le.inner.Close()
				le.inner = nil
				log.Printf("Embedder unloaded after %v idle", timeout)
			}
			le.mu.Unlock()
		case <-le.stopIdle:
			return
		}
	}
}
