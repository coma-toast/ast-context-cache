package dashboard

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
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
	load := sys.HostLoadAverage()
	h.LoadAvgAvailable = load.Available
	h.LoadAvg1 = load.Load1
	h.LoadAvg5 = load.Load5
	h.LoadAvg15 = load.Load15
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
		h.DiskSize = db.FormatFileSize(diskBytes)
	} else {
		h.DiskSize = "-"
	}
	walBytes := db.WalFileBytes()
	h.WalMB = float64(walBytes) / (1024 * 1024)
	h.WalSize = db.FormatFileSize(walBytes)

	ws := watcher.GetStatus()
	if watchers, ok := ws["watchers"].([]map[string]interface{}); ok {
		for _, w := range watchers {
			pp, _ := w["project_path"].(string)
			if projectID != "" && pp != projectID {
				continue
			}
			active, _ := w["active"].(bool)
			meta := projectmeta.Enrich(pp)
			label := meta.Label
			if label == "" {
				label = filepath.Base(pp)
			}
			name := meta.RepoName
			if name == "" {
				name = filepath.Base(pp)
			}
			h.Watchers = append(h.Watchers, components.WatcherInfo{
				ProjectPath: pp,
				Name:        name,
				Label:       label,
				Workspace:   meta.Workspace,
				Active:      active,
			})
		}
	}
	eq := embedqueue.Snapshot()
	h.EmbedQueued = eq.Queued
	h.EmbedPending = eq.Pending
	h.EmbedPendingPeak = eq.PendingPeak
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
	inv := memory.Inventory()
	m.ActiveFacts = inv.ActiveFacts
	m.ActiveProcedures = inv.ActiveProcedures
	m.StructuredMemoryTokens = inv.ActiveTokens
	m.MemoryOrphanCount = inv.OrphanCount
	appendMemoryDocSources(&m, docSourcesPage)
	return m
}
