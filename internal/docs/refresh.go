package docs

import (
	"log"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const quietDocRefreshCooldown = time.Hour

var (
	refreshMu  sync.Mutex
	refreshing = map[int]struct{}{}

	quietRefreshMu   sync.Mutex
	lastQuietRefresh time.Time
	quietRefreshBusy bool
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

// TryQuietRefresh refreshes stale doc caches during a quiet period (async, debounced).
// Only sources older than DocSourceMaxAge are re-fetched.
func TryQuietRefresh(reason string) {
	quietRefreshMu.Lock()
	if quietRefreshBusy {
		quietRefreshMu.Unlock()
		return
	}
	if !lastQuietRefresh.IsZero() && time.Since(lastQuietRefresh) < quietDocRefreshCooldown {
		quietRefreshMu.Unlock()
		return
	}
	stale := countStaleSources()
	if stale == 0 {
		quietRefreshMu.Unlock()
		return
	}
	quietRefreshBusy = true
	lastQuietRefresh = time.Now()
	quietRefreshMu.Unlock()

	go func() {
		defer func() {
			quietRefreshMu.Lock()
			quietRefreshBusy = false
			quietRefreshMu.Unlock()
		}()
		log.Printf("docs: quiet period (%s) — refreshing %d stale source(s)", reason, stale)
		UpdateAllSources()
		notifyDocPanels()
	}()
}

func countStaleSources() int {
	if db.ContextDB == nil {
		return 0
	}
	sources, err := ListSources()
	if err != nil {
		return 0
	}
	n := 0
	for _, s := range sources {
		if SourceNeedsRefresh(s.LastUpdated) {
			n++
		}
	}
	return n
}

// ResetQuietRefreshForTest clears quiet-refresh debounce state (tests only).
func ResetQuietRefreshForTest() {
	quietRefreshMu.Lock()
	lastQuietRefresh = time.Time{}
	quietRefreshBusy = false
	quietRefreshMu.Unlock()
}

func notifyDocPanels() {
	realtime.Notify(realtime.IndexHealth | realtime.Settings)
}
