package docs

import (
	"log"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

var (
	refreshMu  sync.Mutex
	refreshing = map[int]struct{}{}
)

// IsRefreshing reports whether a doc source is being force-fetched from its URL.
func IsRefreshing(id int) bool {
	refreshMu.Lock()
	defer refreshMu.Unlock()
	_, ok := refreshing[id]
	return ok
}

// ForceRefreshSource re-fetches from the source URL regardless of cache age (async).
// Safe to call repeatedly; concurrent refresh for the same id is ignored.
func ForceRefreshSource(id int) {
	refreshMu.Lock()
	if _, ok := refreshing[id]; ok {
		refreshMu.Unlock()
		return
	}
	refreshing[id] = struct{}{}
	refreshMu.Unlock()
	notifyDocPanels()
	go func() {
		if _, err := UpdateSource(id); err != nil {
			log.Printf("doc source %d force refresh: %v", id, err)
		}
		refreshMu.Lock()
		delete(refreshing, id)
		refreshMu.Unlock()
		notifyDocPanels()
	}()
}

func notifyDocPanels() {
	realtime.Notify(realtime.IndexHealth | realtime.Settings)
}
