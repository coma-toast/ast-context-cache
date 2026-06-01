package components

import (
	"fmt"
	"path/filepath"
	"time"
)

func pct(used, cap int) float64 {
	if cap <= 0 {
		return 0
	}
	p := float64(used) / float64(cap) * 100
	if p > 100 {
		return 100
	}
	return p
}

func gaugeLevel(p float64) string {
	switch {
	case p >= 85:
		return "critical"
	case p >= 45:
		return "warn"
	default:
		return "ok"
	}
}

func fillStyle(used, cap int, color string) string {
	return fmt.Sprintf("width:%.1f%%;background:%s", pct(used, cap), color)
}

func ringStyle(used, cap int) string {
	level := gaugeLevel(pct(used, cap))
	return fmt.Sprintf("--ring-pct:%.2f;--ring-level:%s", pct(used, cap), level)
}

func throughputStyle(rate int64) string {
	const max = 80
	p := float64(rate) / max * 100
	if p > 100 {
		p = 100
	}
	return fmt.Sprintf("width:%.1f%%", p)
}

func (h IndexHealth) EmbedPanelBusy() bool {
	return h.EmbedActive > 0 || h.EmbedQueued > 0
}

func (h IndexHealth) EmbedActivityLabel(item EmbedActivityItem) string {
	file := filepath.Base(item.File)
	if file == "" || file == "." {
		file = item.File
	}
	proj := filepath.Base(item.ProjectPath)
	if proj == "" || proj == "." {
		proj = item.ProjectPath
	}
	if proj == "" {
		return file
	}
	if file == "" {
		return proj
	}
	return proj + " · " + file
}

func (h IndexHealth) EmbedActivityTitle(item EmbedActivityItem) string {
	switch {
	case item.ProjectPath != "" && item.File != "":
		return item.ProjectPath + "\n" + item.File
	case item.File != "":
		return item.File
	default:
		return item.ProjectPath
	}
}

func (h IndexHealth) EmbedRecentPreview() []EmbedActivityItem {
	const max = 6
	if len(h.EmbedRecent) <= max {
		return h.EmbedRecent
	}
	return h.EmbedRecent[:max]
}

func (h IndexHealth) queueTotalCap() int {
	return h.EmbedHighCap + h.EmbedLowCap
}

func (h IndexHealth) queueFillPct() float64 {
	return pct(h.EmbedQueued, h.queueTotalCap())
}

func (h IndexHealth) queueLevel() string {
	return gaugeLevel(h.queueFillPct())
}

func (h Health) queueFillPct() float64 {
	return pct(h.QueueQueued, h.QueueHighCap+h.QueueLowCap)
}

func (h Health) queueLevel() string {
	return gaugeLevel(h.queueFillPct())
}

func (h Health) queueMiniStyle() string {
	return fmt.Sprintf("width:%.1f%%", h.queueFillPct())
}

type TodayMeterFill struct {
	WidthPct float64 // bar width: 100 when above avg, else today/avg×100
	AvgPct   float64 // share of bar at daily avg when above avg (avg/today×100)
	AboveAvg bool
	GaugePct float64 // today/avg×100 for level coloring (uncapped)
}

func todayMeterFill(today, total30d int) TodayMeterFill {
	if total30d <= 0 {
		return TodayMeterFill{}
	}
	dailyAvg := float64(total30d) / 30.0
	if dailyAvg <= 0 {
		return TodayMeterFill{}
	}
	ratio := float64(today) / dailyAvg
	gaugePct := ratio * 100
	if gaugePct > 100 {
		return TodayMeterFill{
			WidthPct: 100,
			AvgPct:   dailyAvg / float64(today) * 100,
			AboveAvg: true,
			GaugePct: gaugePct,
		}
	}
	return TodayMeterFill{WidthPct: gaugePct, GaugePct: gaugePct}
}

func todayDurationMeterFill(today, avg float64) TodayMeterFill {
	if avg <= 0 {
		return TodayMeterFill{}
	}
	ratio := today / avg
	gaugePct := ratio * 100
	if gaugePct > 100 {
		return TodayMeterFill{
			WidthPct: 100,
			AvgPct:   avg / today * 100,
			AboveAvg: true,
			GaugePct: gaugePct,
		}
	}
	return TodayMeterFill{WidthPct: gaugePct, GaugePct: gaugePct}
}

func fmtDailyAvgInt(total30d int) string {
	if total30d <= 0 {
		return "0"
	}
	avg := float64(total30d) / 30.0
	if avg >= 100 {
		return fmtInt(int(avg + 0.5))
	}
	if avg == float64(int(avg)) {
		return fmtInt(int(avg))
	}
	return fmt.Sprintf("%.1f", avg)
}

func (s Stats) queriesSublabel() string {
	return fmt.Sprintf("30d: %s · avg/day: %s", fmtInt(s.TotalQueries), fmtDailyAvgInt(s.TotalQueries))
}

func (s Stats) tokensSublabel() string {
	return fmt.Sprintf("30d: %s · avg/day: %s · dedup: %s · vs files: %s", fmtInt(s.TokensSaved), fmtDailyAvgInt(s.TokensSaved), fmtInt(s.DedupTokensSaved), fmtInt(s.SavingsVsFiles))
}

func (s Stats) sessionsSublabel() string {
	return fmt.Sprintf("30d: %s · avg/day: %s · chars: %s", fmtInt(s.Sessions), fmtDailyAvgInt(s.Sessions), fmtInt(s.TotalChars))
}

func (s Stats) virtualSublabel() string {
	quota := ""
	if s.VirtualMaxTokensGlobal > 0 {
		quota = fmt.Sprintf(" · %d/%d notes", s.VirtualNotesCount, s.VirtualMaxNotesGlobal)
	}
	return fmt.Sprintf("%d notes · 30d util: %.0f%% · accessed: %s · orphan: %d · flushed: %s%s",
		s.VirtualNotesCount, s.VirtualUtilPct30d, fmtInt(s.VirtualAccessed30d), s.VirtualOrphanCount, fmtInt(s.VirtualFlushed30d), quota)
}

func (s Stats) virtualInventoryMeter() TodayMeterFill {
	if s.VirtualMaxTokensGlobal <= 0 {
		return todayMeterFill(s.VirtualInventoryTokens, maxInt(s.VirtualStored30d, 1))
	}
	return todayMeterFill(s.VirtualInventoryTokens, s.VirtualMaxTokensGlobal)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (s Stats) todayQueryMeter() TodayMeterFill {
	return todayMeterFill(s.TodayQueries, s.TotalQueries)
}

func (s Stats) todayTokenMeter() TodayMeterFill {
	return todayMeterFill(s.TodayTokens, s.TokensSaved)
}

func (s Stats) todaySessionMeter() TodayMeterFill {
	return todayMeterFill(s.TodaySessions, s.Sessions)
}

func (s Stats) todayDurationMeter() TodayMeterFill {
	return todayDurationMeterFill(s.TodayAvgDurationMs, s.AvgDurationMs)
}

func (s Stats) todayQueryPct() float64 {
	return todayVsDailyAvgPct(s.TodayQueries, s.TotalQueries)
}

func (s Stats) todayTokenPct() float64 {
	return todayVsDailyAvgPct(s.TodayTokens, s.TokensSaved)
}

func (s Stats) todaySessionPct() float64 {
	return todayVsDailyAvgPct(s.TodaySessions, s.Sessions)
}

func (s Stats) todayDurationPct() float64 {
	if s.AvgDurationMs <= 0 {
		return 0
	}
	p := s.TodayAvgDurationMs / s.AvgDurationMs * 100
	if p > 100 {
		return 100
	}
	return p
}

// todayVsDailyAvgPct fills the stat meter: 100% when today matches the rolling 30d daily average.
func todayVsDailyAvgPct(today, total30d int) float64 {
	if total30d <= 0 {
		return 0
	}
	dailyAvg := float64(total30d) / 30.0
	if dailyAvg <= 0 {
		return 0
	}
	p := float64(today) / dailyAvg * 100
	if p > 100 {
		return 100
	}
	return p
}

func (h IndexHealth) WatcherActiveCount() int {
	n := 0
	for _, w := range h.Watchers {
		if w.Active {
			n++
		}
	}
	return n
}

// FormatDocSourceAge returns a compact age string and whether the source is stale (at or past maxAge).
func FormatDocSourceAge(lastUpdated string, maxAge time.Duration) (age string, stale bool) {
	if lastUpdated == "" {
		return "never", true
	}
	t, err := time.Parse(time.RFC3339, lastUpdated)
	if err != nil {
		return "unknown", true
	}
	d := time.Since(t)
	stale = d >= maxAge
	return formatAgeDuration(d), stale
}

func formatAgeDuration(d time.Duration) string {
	if d < time.Minute {
		return "just now"
	}
	days := int(d.Hours() / 24)
	hours := int(d.Hours()) % 24
	if days > 0 {
		if hours > 0 {
			return fmt.Sprintf("%dd %dh", days, hours)
		}
		return fmt.Sprintf("%dd", days)
	}
	if d >= time.Hour {
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
	return fmt.Sprintf("%dm", int(d.Minutes()))
}

func pinnedPct(pinned int) float64 {
	const maxShown = 8.0
	p := float64(pinned) / maxShown * 100
	if p > 100 {
		return 100
	}
	return p
}

func memoryPct(mb float64) float64 {
	const softMax = 512.0
	p := mb / softMax * 100
	if p > 100 {
		return 100
	}
	return p
}

func diskPct(mb float64) float64 {
	const softMax = 2048.0
	p := mb / softMax * 100
	if p > 100 {
		return 100
	}
	return p
}
