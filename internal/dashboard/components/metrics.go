package components

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
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
	return truncateEmbedLabel(embedder.HumanizeEmbedError(d.EmbedderError), 120)
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

func loadAvgCPUs() float64 {
	n := float64(runtime.NumCPU())
	if n <= 0 {
		return 1
	}
	return n
}

func loadAvgPerCore(load float64) float64 {
	return load / loadAvgCPUs()
}

// loadAvgUtilPct is load ÷ cores as a percentage (uncapped; 200% = 2× overloaded).
func loadAvgUtilPct(load float64) float64 {
	return loadAvgPerCore(load) * 100
}

// loadAvgBarWidth caps bar fill at 100% while level uses uncapped utilization.
func loadAvgBarWidth(utilPct float64) float64 {
	if utilPct > 100 {
		return 100
	}
	if utilPct < 0 {
		return 0
	}
	return utilPct
}

func (h IndexHealth) LoadAvgLabel() string {
	cpus := int(loadAvgCPUs())
	return fmt.Sprintf("%.2f× · %.2f× · %.2f× · %dc",
		loadAvgPerCore(h.LoadAvg1),
		loadAvgPerCore(h.LoadAvg5),
		loadAvgPerCore(h.LoadAvg15),
		cpus)
}

func (h IndexHealth) LoadAvgHint() string {
	cpus := int(loadAvgCPUs())
	return fmt.Sprintf("Host load average (1 / 5 / 15 min) ÷ %d cores — 1.0× = fully utilized, >1.0× = overloaded (raw: %.2f · %.2f · %.2f)",
		cpus, h.LoadAvg1, h.LoadAvg5, h.LoadAvg15)
}

func (h IndexHealth) loadAvgBarPct() float64 {
	return loadAvgBarWidth(loadAvgUtilPct(h.LoadAvg1))
}

func (h IndexHealth) loadAvgLevelPct() float64 {
	return loadAvgUtilPct(h.LoadAvg1)
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
	return fmt.Sprintf("%d busy", h.EmbedActivePrimary)
}

func (h IndexHealth) EmbedWorkersWalBadge() string {
	if h.EmbedWorkers == 0 || h.EmbedWorkersEffective >= h.EmbedWorkers {
		return ""
	}
	return fmt.Sprintf("WAL %d/%d", h.EmbedWorkersEffective, h.EmbedWorkers)
}

func (h IndexHealth) EmbedWorkersWalTitle() string {
	if h.EmbedWorkersEffective >= h.EmbedWorkers {
		return ""
	}
	return fmt.Sprintf("SQLite WAL throttled: %d of %d worker goroutines running", h.EmbedWorkersEffective, h.EmbedWorkers)
}

func (h IndexHealth) EmbedAuxWorkersStatus() string {
	if h.EmbedAuxWorkers == 0 {
		return "off"
	}
	if h.EmbedAuxActive > 0 {
		return fmt.Sprintf("%d active", h.EmbedAuxActive)
	}
	return fmt.Sprintf("%d enabled", h.EmbedAuxWorkers)
}

func (h IndexHealth) EmbedAuxWorkerLabel() string {
	if h.EmbedAuxBackend == "" {
		return "Aux workers"
	}
	return fmt.Sprintf("Aux (%s)", h.EmbedAuxBackend)
}

func AuxWorkerMax() int {
	return embedqueue.AuxMaxWorkers()
}

func (h IndexHealth) EmbedderErrorShort() string {
	return truncateEmbedLabel(embedder.HumanizeEmbedError(h.EmbedderError), 120)
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

func workerStripDisplayTotal(target, effective, live int) int {
	raw := target
	if effective > raw {
		raw = effective
	}
	if live > raw {
		return live
	}
	return raw
}

func workerPillClasses(index, active, target, effective, live int) string {
	serverTotal := effective
	serverLive := live
	busy := index < active
	if index >= target && index < serverLive {
		if busy {
			return "worker-pill draining draining-busy"
		}
		return "worker-pill draining"
	}
	if index >= serverTotal && index < target {
		if index < serverLive {
			if busy {
				return "worker-pill busy"
			}
			return "worker-pill idle"
		}
		return "worker-pill pending"
	}
	if busy {
		return "worker-pill busy"
	}
	return "worker-pill idle"
}

func workerStripUsesCompact(maxWorkers int) bool {
	return maxWorkers > 15
}

func workerStripSplitClass(maxWorkers, displayTotal int) bool {
	return !workerStripUsesCompact(maxWorkers) && displayTotal > WorkerStripPerRow
}

func workerStripBusyLabel(active, target int) string {
	return fmt.Sprintf("%d of %d workers busy", min(active, target), target)
}

func workerControlsTitle(active, target, effective, live int) string {
	if target == 0 {
		return "Workers paused — click + to resume"
	}
	if effective < target {
		return fmt.Sprintf("Workers: %d target · %d running (WAL throttled) · %d busy", target, effective, active)
	}
	draining := live - target
	if draining > 0 {
		return fmt.Sprintf("Workers: %d target · %d busy · %d draining", target, active, draining)
	}
	return fmt.Sprintf("Workers: %d of %d busy", active, target)
}

func WorkerControlsTitle(active, target, effective, live int) string {
	return workerControlsTitle(active, target, effective, live)
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
	case "loading":
		return "Starting up"
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
	if h.QueueWorkersLive > h.QueueWorkers {
		return fmt.Sprintf("Workers: %d target · %d busy · %d draining", h.QueueWorkers, h.QueueInFlight, h.QueueWorkersLive-h.QueueWorkers)
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

func (s Stats) kvRepairSublabel() string {
	return fmt.Sprintf("30d: %d repairs · miss: %d · quality: %d · manual: %d · util: %.0f%% · orphans: %d",
		s.KvRepairRepairsTotal30d, s.KvRepairCacheMiss30d, s.KvRepairQuality30d, s.KvRepairManual30d, s.KvRepairUtilPct30d, s.KvRepairOrphans)
}

func (s Stats) kvRepairMeter() TodayMeterFill {
	return todayMeterFill(s.KvRepairRepairsTotal30d, maxInt(s.KvRepairArchivesStored30d, 1))
}

func (s Stats) kvRepairArchivesSublabel() string {
	return fmt.Sprintf("30d stored: %s · tokens repaired: %s · accessed today: %s",
		fmtInt(s.KvRepairArchivesStored30d), fmtInt(s.KvRepairTokensRepaired30d), fmtInt(s.VirtualTodayAccessed))
}

func (s Stats) kvRepairArchivesMeter() TodayMeterFill {
	return todayMeterFill(s.KvRepairArchivesActive, maxInt(s.KvRepairArchivesStored30d, 1))
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
	if h.WALMaintenanceActive && h.WALWalStartBytes > 0 {
		start := db.FormatFileSize(h.WALWalStartBytes)
		cur := db.FormatFileSize(h.WALWalCurrentBytes)
		if cur != start {
			return h.DiskSize + " · WAL " + start + " → " + cur
		}
		return h.DiskSize + " · WAL " + cur + " · compacting"
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
