package dashboard

import (
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

func initRealtimeBridge() {
	realtime.SetHandler(flushRefreshBroadcast)
	startLiveRefresh()
}

func flushRefreshBroadcast(mask realtime.Reason) {
	if hub == nil {
		return
	}
	if mask&realtime.IndexHealth != 0 {
		invalidateIndexHealthCache()
	}
	indexBlocked := db.WALMaintenanceActive() || db.IndexReadQuiesced()
	var panels []string
	for _, p := range dashboardPanels {
		if !panelMatchesMask(p.name, mask) {
			continue
		}
		if indexBlocked && panelUsesIndexDB(p.name) {
			continue
		}
		panels = append(panels, p.name)
	}
	if len(panels) == 0 {
		return
	}
	broadcastRefresh(panels)
}

func panelUsesIndexDB(name string) bool {
	switch name {
	case "symbol-chart", "language-chart", "import-chart", "memory", "settings":
		return true
	default:
		return false
	}
}

func panelMatchesMask(name string, mask realtime.Reason) bool {
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

func startLiveRefresh() {
	go func() {
		fast := time.NewTicker(2 * time.Second)
		slow := time.NewTicker(10 * time.Second)
		defer fast.Stop()
		defer slow.Stop()
		for {
			select {
			case <-fast.C:
				if !db.PoolsReady() || db.WALMaintenanceActive() {
					continue
				}
				invalidateIndexHealthCache()
				flushRefreshBroadcast(realtime.IndexHealth | realtime.HealthBar)
			case <-slow.C:
				if !db.PoolsReady() || db.WALMaintenanceActive() {
					continue
				}
				flushRefreshBroadcast(realtime.Stats)
			}
		}
	}()
}

func invalidatePartialCache(_ string) {}
