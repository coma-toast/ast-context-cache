package components

import "fmt"

type MemoryData struct {
	FilteredProject           string
	TotalSymbols              int
	TotalVectors              int
	VectorMemMB               float64
	VirtualInventoryTokens    int
	VirtualNotesCount         int
	VirtualUtilPct30d         float64
	VirtualOrphanCount        int
	VirtualFlushed30d         int
	VirtualStored30d          int
	VirtualAccessed30d        int
	VirtualTodayStored        int
	VirtualTodayAccessed      int
	VirtualMaxNotesGlobal     int
	VirtualMaxTokensGlobal    int
	KvRepairArchivesActive    int
	KvRepairArchivesStored30d int
	KvRepairRepairsTotal30d   int
	KvRepairUtilPct30d        float64
	KvRepairOrphans           int
	KvRepairTokensRepaired30d int
	KvRepairCacheMiss30d      int
	KvRepairQuality30d        int
	KvRepairManual30d         int
	KvRepairTodayRepairs      int
	ActiveFacts               int
	ActiveProcedures          int
	StructuredMemoryTokens    int
	MemoryOrphanCount         int
	DocSources                []IndexDocSource
	DocSourcesTotal           int
	DocSourcesPage            int
	DocSourcesPerPage         int
}

func (m MemoryData) virtualSublabel() string {
	quota := ""
	if m.VirtualMaxTokensGlobal > 0 {
		quota = fmt.Sprintf(" · %d/%d notes", m.VirtualNotesCount, m.VirtualMaxNotesGlobal)
	}
	return fmt.Sprintf("%d notes · 30d util: %.0f%% · accessed: %s · orphan: %d · flushed: %s%s",
		m.VirtualNotesCount, m.VirtualUtilPct30d, fmtInt(m.VirtualAccessed30d), m.VirtualOrphanCount, fmtInt(m.VirtualFlushed30d), quota)
}

func (m MemoryData) virtualInventoryMeter() TodayMeterFill {
	if m.VirtualMaxTokensGlobal <= 0 {
		return todayMeterFill(m.VirtualInventoryTokens, maxInt(m.VirtualStored30d, 1))
	}
	return todayMeterFill(m.VirtualInventoryTokens, m.VirtualMaxTokensGlobal)
}

func (m MemoryData) kvRepairSublabel() string {
	return fmt.Sprintf("30d: %d repairs · miss: %d · quality: %d · manual: %d · util: %.0f%% · orphans: %d",
		m.KvRepairRepairsTotal30d, m.KvRepairCacheMiss30d, m.KvRepairQuality30d, m.KvRepairManual30d, m.KvRepairUtilPct30d, m.KvRepairOrphans)
}

func (m MemoryData) kvRepairMeter() TodayMeterFill {
	return todayMeterFill(m.KvRepairRepairsTotal30d, maxInt(m.KvRepairArchivesStored30d, 1))
}
