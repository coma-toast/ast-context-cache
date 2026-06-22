package components

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
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

func (d SettingsData) EmbedActiveStatusLabel() string {
	switch d.EmbedderState {
	case "error":
		return "Unavailable"
	case "degraded":
		return "Degraded"
	default:
		return "Running"
	}
}

func (d SettingsData) EmbedderErrorShort() string {
	return truncateEmbedLabel(d.EmbedderError, 120)
}

func (h IndexHealth) ShowEmbedDismiss() bool {
	if h.EmbedderState == "degraded" {
		return true
	}
	return h.EmbedderState == "error" && h.EmbedPanelBusy()
}

func (d SettingsData) ShowEmbedDismiss() bool {
	if d.EmbedderState == "degraded" {
		return true
	}
	return d.EmbedderState == "error" && d.EmbedActive > 0
}

func (d SettingsData) EmbedderErrorHeadline() string {
	switch d.EmbedderState {
	case "error":
		return "Embedder unreachable"
	case "degraded":
		return "Recent embed errors"
	default:
		return ""
	}
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

func cpuGaugePct(p float64) float64 {
	if p > 100 {
		return 100
	}
	return p
}

func loadAvgPct(load1 float64) float64 {
	n := float64(runtime.NumCPU())
	if n <= 0 {
		n = 1
	}
	p := load1 / n * 100
	if p > 100 {
		return 100
	}
	return p
}

func (h IndexHealth) LoadAvgLabel() string {
	return fmt.Sprintf("%.2f · %.2f · %.2f", h.LoadAvg1, h.LoadAvg5, h.LoadAvg15)
}

func (h IndexHealth) LoadAvgHint() string {
	return fmt.Sprintf("Host load averages (1 / 5 / 15 min) — gauge vs %d CPU cores", runtime.NumCPU())
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

func (h IndexHealth) EmbedWorkersStatus() string {
	if h.EmbedWorkers == 0 {
		return "paused"
	}
	return fmt.Sprintf("%d active", h.EmbedActive)
}

func (h IndexHealth) EmbedderErrorShort() string {
	if len(h.EmbedderError) <= 120 {
		return h.EmbedderError
	}
	return h.EmbedderError[:117] + "..."
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

const (
	MinEmbedWorkers       = 0
	WorkerStripPerRow     = 5
	WorkerStripMaxRows    = 4
	WorkerStripVisibleMax = WorkerStripPerRow * WorkerStripMaxRows
)

// EmbedWorkerMax is the configured upper limit for embed worker goroutines.
func EmbedWorkerMax() int {
	return embedqueue.MaxWorkers()
}

func workerStripDotCount(total int) (dots int, ellipsis bool) {
	if total > WorkerStripVisibleMax {
		return WorkerStripVisibleMax - 1, true
	}
	return total, false
}

func workerStripUsesCompact(maxWorkers int) bool {
	return maxWorkers > 15
}

func WorkerControlsTitle(active, total int) string {
	if total == 0 {
		return "Workers paused — click + to resume"
	}
	return fmt.Sprintf("Workers: %d of %d busy", active, total)
}

func pendingRingCap(pending, peak int) int {
	if pending <= 0 {
		return 1
	}
	cap := peak
	if cap < pending {
		cap = pending
	}
	if cap < 1 {
		return 1
	}
	return cap
}

func (h IndexHealth) pendingRingCap() int {
	return pendingRingCap(h.EmbedPending, h.EmbedPendingPeak)
}

func (h IndexHealth) queueRingCap() int {
	return h.queueTotalCap()
}

func (h IndexHealth) queueTotalCap() int {
	return h.EmbedHighCap + h.EmbedLowCap
}

func (h IndexHealth) EmbedderErrorHeadline() string {
	switch h.EmbedderState {
	case "error":
		return "Embedder unreachable"
	case "degraded":
		return "Recent embed errors"
	default:
		return ""
	}
}

func (h IndexHealth) ShowEmbedInSyncBadge() bool {
	return h.EmbedInSync && h.EmbedderState != "error" && h.EmbedderState != "degraded"
}

func (h IndexHealth) ShowEmbedBacklogHint() bool {
	if h.EmbedPanelBusy() || h.EmbedThroughput > 0 {
		return false
	}
	return (h.EmbedderState == "error" || h.EmbedderState == "degraded") && h.EmbedPending > 0
}

func (h IndexHealth) queueFillPct() float64 {
	return pct(h.EmbedQueued, h.queueRingCap())
}

func (h IndexHealth) pendingFillPct() float64 {
	return pct(h.EmbedPending, h.pendingRingCap())
}

func (h IndexHealth) queueLevel() string {
	return gaugeLevel(h.queueFillPct())
}

func (h Health) queueTotalCap() int {
	return h.QueueHighCap + h.QueueLowCap
}

func (h Health) queueTitle() string {
	return fmt.Sprintf("Queue %d / %d (priority %d + background %d)",
		h.QueueQueued, h.queueTotalCap(), h.QueueHighCap, h.QueueLowCap)
}

func (h Health) workersTitle() string {
	if h.QueueWorkers == 0 {
		return "Workers paused (0) — use + on embeddings card to resume"
	}
	return fmt.Sprintf("Workers: %d of %d busy", h.QueueInFlight, h.QueueWorkers)
}

func (h Health) pendingTitle() string {
	msg := fmt.Sprintf("Pending %d — files awaiting retry after embed failure", h.QueuePending)
	if (h.EmbedderState == "error" || h.EmbedderState == "degraded") && h.QueuePending > 0 {
		msg += " · backlog will drain when embedder is healthy"
	}
	return msg
}

func (h Health) pendingRingCap() int {
	return pendingRingCap(h.QueuePending, h.QueuePendingPeak)
}

func (h Health) queueFillPct() float64 {
	return pct(h.QueueQueued, h.queueTotalCap())
}

func (h Health) pendingFillPct() float64 {
	return pct(h.QueuePending, h.pendingRingCap())
}

func (h Health) queueLevel() string {
	return gaugeLevel(h.queueFillPct())
}

func (h Health) pendingLevel() string {
	return gaugeLevel(h.pendingFillPct())
}

func (h Health) queueMiniStyle() string {
	return fmt.Sprintf("width:%.1f%%", h.queueFillPct())
}

func (h Health) pendingMiniStyle() string {
	return fmt.Sprintf("width:%.1f%%", h.pendingFillPct())
}

type TodayMeterFill struct {
	OverlapPct float64 // min(day, avg) as share of max(day, avg)
	DayOnlyPct float64 // day above avg (red)
	AvgOnlyPct float64 // avg above day (blue)
	GaugePct   float64 // day/avg×100 (uncapped, for labels)
}

func meterFillSegments(day, avg float64) TodayMeterFill {
	if avg <= 0 {
		return TodayMeterFill{}
	}
	max := day
	if avg > max {
		max = avg
	}
	overlap := day
	if avg < overlap {
		overlap = avg
	}
	dayOnly := day - overlap
	if dayOnly < 0 {
		dayOnly = 0
	}
	avgOnly := avg - overlap
	if avgOnly < 0 {
		avgOnly = 0
	}
	return TodayMeterFill{
		OverlapPct: overlap / max * 100,
		DayOnlyPct: dayOnly / max * 100,
		AvgOnlyPct: avgOnly / max * 100,
		GaugePct:   day / avg * 100,
	}
}

func todayMeterFill(today, total30d int) TodayMeterFill {
	if total30d <= 0 {
		return TodayMeterFill{}
	}
	return meterFillSegments(float64(today), float64(total30d)/30.0)
}

func todayDurationMeterFill(today, avg float64) TodayMeterFill {
	return meterFillSegments(today, avg)
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

func (h IndexHealth) DiskSizeLabel() string {
	if h.DiskSize == "-" {
		return "-"
	}
	if h.WalSize == "" || h.WalSize == "0 B" {
		return h.DiskSize
	}
	return h.DiskSize + " · WAL " + h.WalSize
}

func (h IndexHealth) DiskFootprintPct() float64 {
	return diskPct(h.DiskMB + h.WalMB)
}

func (h IndexHealth) DiskHint() string {
	return "SQLite usage.db + WAL journal (usage.db-wal) on disk"
}

func diskIOPct(mbps float64) float64 {
	const softMax = 200.0
	p := mbps / softMax * 100
	if p > 100 {
		return 100
	}
	return p
}

func formatDiskIO(mbps float64) string {
	if mbps <= 0 {
		return "0 MB/s"
	}
	if mbps >= 1024 {
		return fmt.Sprintf("%.1f GB/s", mbps/1024)
	}
	if mbps >= 10 {
		return fmt.Sprintf("%.0f MB/s", mbps)
	}
	return fmt.Sprintf("%.1f MB/s", mbps)
}

func ssdSmartClass(status string) string {
	s := strings.ToLower(strings.TrimSpace(status))
	switch {
	case strings.Contains(s, "verified"), strings.Contains(s, "ok"), strings.Contains(s, "normal"):
		return "ssd-smart-ok"
	case strings.Contains(s, "fail"), strings.Contains(s, "error"), strings.Contains(s, "critical"):
		return "ssd-smart-bad"
	default:
		return "ssd-smart-unknown"
	}
}

func (h IndexHealth) SSDModelShort() string {
	if h.SSDModel == "" {
		return "-"
	}
	if len(h.SSDModel) <= 28 {
		return h.SSDModel
	}
	return h.SSDModel[:25] + "..."
}

func (h IndexHealth) SSDTrimLabel() string {
	if h.SSDTrim {
		return "Yes"
	}
	return "No"
}

func (h IndexHealth) SSDWearKnown() bool {
	return h.SSDWearUsedPct >= 0
}

func (h IndexHealth) SSDSpareKnown() bool {
	return h.SSDSparePct >= 0
}

func (h IndexHealth) SSDDataWrittenKnown() bool {
	return h.SSDDataWrittenTB >= 0
}

func (h IndexHealth) SSDTemperatureKnown() bool {
	return h.SSDTemperatureC >= 0
}

func (h IndexHealth) SSDWearLabel() string {
	if !h.SSDWearKnown() {
		return "-"
	}
	return fmt.Sprintf("%d%% used", h.SSDWearUsedPct)
}

func (h IndexHealth) SSDSpareLabel() string {
	if !h.SSDSpareKnown() {
		return "-"
	}
	return fmt.Sprintf("%d%%", h.SSDSparePct)
}

func (h IndexHealth) SSDDataWrittenLabel() string {
	if !h.SSDDataWrittenKnown() {
		return "-"
	}
	if h.SSDDataWrittenTB >= 100 {
		return fmt.Sprintf("%.0f TB", h.SSDDataWrittenTB)
	}
	return fmt.Sprintf("%.1f TB", h.SSDDataWrittenTB)
}

func (h IndexHealth) SSDTemperatureLabel() string {
	if !h.SSDTemperatureKnown() {
		return "-"
	}
	return fmt.Sprintf("%.1f °C", h.SSDTemperatureC)
}

func ssdWearColor(pct int) string {
	switch {
	case pct >= 85:
		return "#f85149"
	case pct >= 50:
		return "#f0883e"
	default:
		return "#3fb950"
	}
}
