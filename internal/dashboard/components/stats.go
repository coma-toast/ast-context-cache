package components

import "fmt"

type Stats struct {
	TotalQueries              int
	TodayQueries              int
	TokensSaved               int
	TodayTokens               int
	DedupTokensSaved          int
	SavingsVsFiles            int
	SymbolBaseline            int
	AvgDurationMs             float64
	TodayAvgDurationMs        float64
	Sessions                  int
	TodaySessions             int
	TotalChars                int
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
}

func fmtInt(n int) string {
	if n == 0 {
		return "0"
	}
	s := fmt.Sprintf("%d", n)
	if n < 1000 {
		return s
	}
	result := ""
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			result += ","
		}
		result += string(c)
	}
	return result
}

func fmtInt64(n int64) string {
	if n == 0 {
		return "0"
	}
	s := fmt.Sprintf("%d", n)
	if n < 1000 {
		return s
	}
	result := ""
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			result += ","
		}
		result += string(c)
	}
	return result
}

func fmtBool(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
