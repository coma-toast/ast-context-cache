package dashboard

import (
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

var (
	partialLast   sync.Map // target string -> last HTML
	notifyMu      sync.Mutex
	pendingMask   realtime.Reason
	notifyTimer   *time.Timer
	debounceDelay = 150 * time.Millisecond
)

func initRealtimeBridge() {
	realtime.SetHandler(handleRealtimeNotify)
}

func handleRealtimeNotify(mask realtime.Reason) {
	notifyMu.Lock()
	defer notifyMu.Unlock()
	pendingMask |= mask
	if notifyTimer != nil {
		notifyTimer.Stop()
	}
	notifyTimer = time.AfterFunc(debounceDelay, func() {
		notifyMu.Lock()
		m := pendingMask
		pendingMask = 0
		notifyMu.Unlock()
		flushPartialBroadcast(m)
	})
}

func flushPartialBroadcast(mask realtime.Reason) {
	if hub == nil {
		return
	}
	for _, p := range dashboardPartials {
		if !partialMatchesMask(p.name, mask) {
			continue
		}
		html := p.render()
		if last, ok := partialLast.Load(p.target); ok && last.(string) == html {
			continue
		}
		partialLast.Store(p.target, html)
		hub.broadcast <- wsMsg{
			Type:      "partial",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]string{
				"target": p.target,
				"html":   html,
			},
		}
	}
}

func invalidatePartialCache(target string) {
	partialLast.Delete(target)
}

func partialMatchesMask(name string, mask realtime.Reason) bool {
	switch name {
	case "index-health":
		return mask&realtime.IndexHealth != 0
	case "memory":
		return mask&(realtime.Stats|realtime.IndexHealth|realtime.Settings) != 0
	case "health-bar":
		return mask&realtime.HealthBar != 0
	case "stats":
		return mask&realtime.Stats != 0
	case "recent":
		return mask&realtime.Recent != 0
	case "tool-chart":
		return mask&realtime.ToolChart != 0
	case "symbol-chart":
		return mask&realtime.SymbolChart != 0
	case "language-chart":
		return mask&realtime.LanguageChart != 0
	case "import-chart":
		return mask&realtime.ImportChart != 0
	case "settings":
		return mask&realtime.Settings != 0
	default:
		return false
	}
}
