// Package realtime decouples dashboard WebSocket updates from producers (indexer, embed queue, db).
// Producers call Notify; dashboard registers a handler that re-renders and broadcasts partials.
package realtime

import (
	"sync"
	"time"
)

// Reason is a bitmask of dashboard panels that may need refresh.
type Reason uint32

const (
	IndexHealth Reason = 1 << iota
	HealthBar
	Stats
	Recent
	ToolChart
	SymbolChart
	LanguageChart
	ImportChart
	Settings
)

// Composite masks for common events.
const (
	IndexCommitted  = IndexHealth | HealthBar | SymbolChart | LanguageChart | ImportChart
	EmbedFinished   = IndexHealth | HealthBar
	WatchersChanged = IndexHealth
	QueryLogged     = Stats | Recent | ToolChart
	SettingsChanged = Settings
)

const debounceDelay = 150 * time.Millisecond

var (
	handlerMu sync.RWMutex
	handler   func(Reason)

	timerMu sync.Mutex
	pending Reason
	timer   *time.Timer
)

// SetHandler registers the dashboard callback (typically from dashboard init).
func SetHandler(h func(Reason)) {
	handlerMu.Lock()
	handler = h
	handlerMu.Unlock()
}

// Notify schedules a debounced refresh for the given panels (OR-combined with pending).
func Notify(r Reason) {
	if r == 0 {
		return
	}
	timerMu.Lock()
	pending |= r
	if timer == nil {
		timer = time.AfterFunc(debounceDelay, fire)
	} else {
		timer.Reset(debounceDelay)
	}
	timerMu.Unlock()
}

func fire() {
	timerMu.Lock()
	mask := pending
	pending = 0
	timer = nil
	timerMu.Unlock()
	if mask == 0 {
		return
	}
	handlerMu.RLock()
	h := handler
	handlerMu.RUnlock()
	if h != nil {
		h(mask)
	}
}
