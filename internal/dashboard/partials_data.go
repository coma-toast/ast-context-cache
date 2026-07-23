package dashboard

import (
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

var (
	indexHealthCacheMu sync.Mutex
	indexHealthCache   struct {
		at  time.Time
		pid string
		h   components.IndexHealth
	}
	indexHealthCacheTTL = 2 * time.Second
)

func indexHealthCacheTTLForRefresh() time.Duration {
	if eq := embedqueue.Snapshot(); eq.InFlight > 0 || eq.Queued > 0 || eq.Pending > 0 || db.WALMaintenanceActive() {
		return 0
	}
	return indexHealthCacheTTL
}

func buildIndexHealth(projectID string) components.IndexHealth {
	if db.WALMaintenanceActive() {
		indexHealthCacheMu.Lock()
		if !indexHealthCache.at.IsZero() && indexHealthCache.pid == projectID {
			h := applyLiveHealthSignals(indexHealthCache.h)
			indexHealthCacheMu.Unlock()
			return h
		}
		indexHealthCacheMu.Unlock()
	}
	indexHealthCacheMu.Lock()
	ttl := indexHealthCacheTTLForRefresh()
	if ttl > 0 && !indexHealthCache.at.IsZero() && time.Since(indexHealthCache.at) < ttl && indexHealthCache.pid == projectID {
		h := indexHealthCache.h
		indexHealthCacheMu.Unlock()
		return h
	}
	indexHealthCacheMu.Unlock()
	h := buildIndexHealthFresh(projectID)
	indexHealthCacheMu.Lock()
	indexHealthCache.at = time.Now()
	indexHealthCache.pid = projectID
	indexHealthCache.h = h
	indexHealthCacheMu.Unlock()
	return h
}

func invalidateIndexHealthCache() {
	indexHealthCacheMu.Lock()
	indexHealthCache.at = time.Time{}
	indexHealthCacheMu.Unlock()
}

func applyLiveHealthSignals(h components.IndexHealth) components.IndexHealth {
	walSnap := db.GetWALSnapshot()
	h.WALMaintenanceActive = walSnap.Active
	h.WALMaintenancePhase = string(walSnap.Phase)
	h.WALMaintenanceMode = walSnap.Mode
	h.WALMaintenanceReason = walSnap.Reason
	h.WALMaintenanceStarted = walSnap.StartedAt
	h.WALWalStartBytes = walSnap.WalStartBytes
	h.WALWalCurrentBytes = walSnap.WalCurrentBytes
	h.WALBusyStreak = walSnap.BusyStreak
	h.WALInFlight = walSnap.InFlight
	h.WALLastBusy = walSnap.LastBusy
	h.WALPressure = db.WalPressure()
	eq := embedqueue.Snapshot()
	h.EmbedQueued = eq.Queued
	h.EmbedPending = eq.Pending
	h.EmbedPendingPeak = eq.PendingPeak
	h.EmbedFailed = eq.Failed
	h.EmbedHighQueued = eq.HighUsed
	h.EmbedLowQueued = eq.LowUsed
	h.EmbedActive = int(eq.InFlight)
	h.EmbedWorkers = eq.Workers
	h.EmbedWorkersEffective = eq.WorkersEffective
	h.EmbedWorkersLive = eq.WorkersLive
	h.EmbedAuxWorkers = eq.AuxWorkers
	h.EmbedAuxWorkersEffective = eq.AuxWorkersEffective
	h.EmbedAuxWorkersLive = eq.AuxWorkersLive
	h.EmbedComplete = eq.Completed
	h.EmbedThroughput = eq.Throughput
	h.LastAutoRecoverUnix = eq.LastAutoRecoverUnix
	return h
}

func buildIndexHealthFresh(projectID string) components.IndexHealth {
	h := components.IndexHealth{}
	if db.IndexDB == nil {
		applyActiveEmbedder(&h)
		return h
	}
	if projectID != "" {
		db.IndexDB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", projectID).Scan(&h.TotalSymbols, &h.TotalFiles)
		db.IndexDB.QueryRow("SELECT COUNT(*) FROM edges WHERE project_path = ?", projectID).Scan(&h.TotalEdges)
	} else {
		db.IndexDB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&h.TotalSymbols, &h.TotalFiles)
		db.IndexDB.QueryRow("SELECT COUNT(*) FROM edges").Scan(&h.TotalEdges)
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

	diskBytes := db.TotalDBBytes()
	if diskBytes > 0 {
		h.DiskMB = float64(diskBytes) / (1024 * 1024)
		h.DiskSize = db.FormatFileSize(diskBytes)
	} else {
		h.DiskSize = "-"
	}
	walBytes := db.IndexWalBytes()
	h.WalMB = float64(walBytes) / (1024 * 1024)
	h.WalSize = db.FormatFileSize(walBytes)

	ws := watcher.GetStatus()
	if watchers, ok := ws["watchers"].([]map[string]interface{}); ok {
		for _, w := range watchers {
			pp, _ := w["project_path"].(string)
			if projectID != "" && pp != projectID {
				continue
			}
			if projectmeta.IsExcluded(pp) {
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
			linked, _ := projectlinks.Links(pp)
			h.Watchers = append(h.Watchers, components.WatcherInfo{
				ProjectPath: pp,
				Name:        name,
				Label:       label,
				Workspace:   meta.Workspace,
				Active:      active,
				LinkedCount: len(linked),
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
	h.EmbedWorkersEffective = eq.WorkersEffective
	h.EmbedWorkersLive = eq.WorkersLive
	h.EmbedWorkerMax = embedqueue.MaxWorkers()
	h.EmbedAuxWorkers = eq.AuxWorkers
	h.EmbedAuxWorkersEffective = eq.AuxWorkersEffective
	h.EmbedAuxWorkersLive = eq.AuxWorkersLive
	h.EmbedAuxWorkerMax = embedqueue.AuxMaxWorkers()
	h.EmbedAuxBackend, h.EmbedAuxModel = embedder.AuxSnapshot()
	if h.EmbedAuxBackend == "" {
		h.EmbedAuxBackend = embedder.AuxBackend()
	}
	h.EmbedAuxEnabled = h.EmbedAuxBackend != "" && !embedder.AuxSharesPrimary()
	h.EmbedComplete = eq.Completed
	h.EmbedThroughput = eq.Throughput
	h.LastAutoRecoverUnix = eq.LastAutoRecoverUnix
	h.PinnedCount = db.PinnedProjectCount()
	h.FilteredProject = projectID
	walSnap := db.GetWALSnapshot()
	h.WALMaintenanceActive = walSnap.Active
	h.WALMaintenancePhase = string(walSnap.Phase)
	h.WALMaintenanceMode = walSnap.Mode
	h.WALMaintenanceReason = walSnap.Reason
	h.WALMaintenanceStarted = walSnap.StartedAt
	h.WALWalStartBytes = walSnap.WalStartBytes
	h.WALWalCurrentBytes = walSnap.WalCurrentBytes
	h.WALBusyStreak = walSnap.BusyStreak
	h.WALInFlight = walSnap.InFlight
	h.WALLastBusy = walSnap.LastBusy
	h.WALPressure = db.WalPressure()
	applyActiveEmbedder(&h)
	return h
}

func buildMemory(projectID string, docSourcesPage int) components.MemoryData {
	m := components.MemoryData{FilteredProject: projectID}
	if db.IndexDB == nil {
		return m
	}
	if projectID != "" {
		db.IndexDB.QueryRow("SELECT COUNT(*) FROM symbols WHERE project_path = ?", projectID).Scan(&m.TotalSymbols)
	} else {
		db.IndexDB.QueryRow("SELECT COUNT(*) FROM symbols").Scan(&m.TotalSymbols)
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
	m.KvRepairArchivesActive = s.KvRepairArchivesActive
	m.KvRepairArchivesStored30d = s.KvRepairArchivesStored30d
	m.KvRepairRepairsTotal30d = s.KvRepairRepairsTotal30d
	m.KvRepairUtilPct30d = s.KvRepairUtilPct30d
	m.KvRepairOrphans = s.KvRepairOrphans
	m.KvRepairTokensRepaired30d = s.KvRepairTokensRepaired30d
	m.KvRepairCacheMiss30d = s.KvRepairCacheMiss30d
	m.KvRepairQuality30d = s.KvRepairQuality30d
	m.KvRepairManual30d = s.KvRepairManual30d
	m.KvRepairTodayRepairs = s.KvRepairTodayRepairs
	inv := memory.Inventory()
	m.ActiveFacts = inv.ActiveFacts
	m.ActiveProcedures = inv.ActiveProcedures
	m.StructuredMemoryTokens = inv.ActiveTokens
	m.MemoryOrphanCount = inv.OrphanCount
	appendMemoryDocSources(&m, docSourcesPage)
	return m
}
