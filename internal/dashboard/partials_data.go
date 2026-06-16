package dashboard

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
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
	diskIO := sys.DiskIORates()
	h.DiskReadMBps = diskIO.ReadMBps
	h.DiskWriteMBps = diskIO.WriteMBps
	ssd := sys.SSDHealthInfo()
	h.SSDAvailable = ssd.Available
	h.SSDModel = ssd.Model
	h.SSDSmartStatus = ssd.SmartStatus
	h.SSDProtocol = ssd.Protocol
	h.SSDCapacity = ssd.Capacity
	h.SSDSolidState = ssd.SolidState
	h.SSDTrim = ssd.TrimSupport
	h.SSDWearUsedPct = ssd.WearUsedPct
	h.SSDSparePct = ssd.SparePct
	h.SSDDataWrittenTB = ssd.DataWrittenTB
	h.SSDTemperatureC = ssd.TemperatureC

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
	eq := embedqueue.Snapshot()
	h.EmbedQueued = eq.Queued
	h.EmbedPending = eq.Pending
	h.EmbedFailed = eq.Failed
	h.EmbedHighQueued = eq.HighUsed
	h.EmbedLowQueued = eq.LowUsed
	h.EmbedHighCap = eq.HighCap
	h.EmbedLowCap = eq.LowCap
	h.EmbedActive = int(eq.InFlight)
	h.EmbedWorkers = eq.Workers
	h.EmbedWorkersLive = eq.WorkersLive
	h.EmbedComplete = eq.Completed
	h.EmbedThroughput = eq.Throughput
	h.PinnedCount = db.PinnedProjectCount()
	h.FilteredProject = projectID
	applyActiveEmbedder(&h)
	return h
}

func buildMemory(projectID string, docSourcesPage int) components.MemoryData {
	m := components.MemoryData{FilteredProject: projectID}
	if projectID != "" {
		db.DB.QueryRow("SELECT COUNT(*) FROM symbols WHERE project_path = ?", projectID).Scan(&m.TotalSymbols)
	} else {
		db.DB.QueryRow("SELECT COUNT(*) FROM symbols").Scan(&m.TotalSymbols)
	}
	m.TotalVectors = search.Cache.Count(projectID)
	m.VectorMemMB = search.Cache.MemoryMB()
	var s components.Stats
	fillVirtualContextStats(&s, projectID)
	m.VirtualInventoryTokens = s.VirtualInventoryTokens
	m.VirtualNotesCount = s.VirtualNotesCount
	m.VirtualUtilPct30d = s.VirtualUtilPct30d
	m.VirtualOrphanCount = s.VirtualOrphanCount
	m.VirtualFlushed30d = s.VirtualFlushed30d
	m.VirtualStored30d = s.VirtualStored30d
	m.VirtualAccessed30d = s.VirtualAccessed30d
	m.VirtualTodayStored = s.VirtualTodayStored
	m.VirtualTodayAccessed = s.VirtualTodayAccessed
	m.VirtualMaxNotesGlobal = s.VirtualMaxNotesGlobal
	m.VirtualMaxTokensGlobal = s.VirtualMaxTokensGlobal
	appendMemoryDocSources(&m, docSourcesPage)
	return m
}
