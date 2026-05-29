package dashboard

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

func buildIndexHealth(projectID string) components.IndexHealth {
	h := components.IndexHealth{}
	if projectID != "" {
		db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", projectID).Scan(&h.TotalSymbols, &h.TotalFiles)
		db.DB.QueryRow("SELECT COUNT(*) FROM edges WHERE project_path = ?", projectID).Scan(&h.TotalEdges)
	} else {
		db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&h.TotalSymbols, &h.TotalFiles)
		db.DB.QueryRow("SELECT COUNT(*) FROM edges").Scan(&h.TotalEdges)
	}
	h.TotalVectors = search.Cache.Count(projectID)
	h.VectorMemMB = search.Cache.MemoryMB()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	h.MemoryMB = float64(memStats.Alloc) / (1024 * 1024)
	h.CPUPercent = sys.ProcessCPUPercent()

	dbPath := db.GetDBPath()
	if fi, err := os.Stat(dbPath); err == nil {
		diskBytes := fi.Size()
		h.DiskMB = float64(diskBytes) / (1024 * 1024)
		switch {
		case diskBytes >= 1024*1024*1024:
			h.DiskSize = fmt.Sprintf("%.2f GB", float64(diskBytes)/(1024*1024*1024))
		case diskBytes >= 1024*1024:
			h.DiskSize = fmt.Sprintf("%.1f MB", float64(diskBytes)/(1024*1024))
		case diskBytes >= 1024:
			h.DiskSize = fmt.Sprintf("%d KB", diskBytes/1024)
		default:
			h.DiskSize = fmt.Sprintf("%d B", diskBytes)
		}
	} else {
		h.DiskSize = "-"
	}

	ws := watcher.GetStatus()
	if watchers, ok := ws["watchers"].([]map[string]interface{}); ok {
		for _, w := range watchers {
			pp, _ := w["project_path"].(string)
			if projectID != "" && pp != projectID {
				continue
			}
			active, _ := w["active"].(bool)
			h.Watchers = append(h.Watchers, components.WatcherInfo{
				ProjectPath: pp,
				Name:        filepath.Base(pp),
				Active:      active,
			})
		}
	}
	if sources, err := docs.ListSources(); err == nil {
		for _, s := range sources {
			age, stale := components.FormatDocSourceAge(s.LastUpdated, docs.DocSourceMaxAge)
			h.DocSources = append(h.DocSources, components.IndexDocSource{
				ID:         s.ID,
				Name:       s.Name,
				Type:       s.Type,
				URL:        s.URL,
				Age:        age,
				Stale:      stale,
				Refreshing: docs.IsRefreshing(s.ID),
			})
		}
	}
	eq := embedqueue.Snapshot()
	h.EmbedQueued = eq.Queued
	h.EmbedHighQueued = eq.HighUsed
	h.EmbedLowQueued = eq.LowUsed
	h.EmbedHighCap = eq.HighCap
	h.EmbedLowCap = eq.LowCap
	h.EmbedActive = int(eq.InFlight)
	h.EmbedWorkers = eq.Workers
	h.EmbedComplete = eq.Completed
	h.EmbedThroughput = eq.Throughput
	h.PinnedCount = db.PinnedProjectCount()
	h.FilteredProject = projectID
	applyActiveEmbedder(&h)
	return h
}
