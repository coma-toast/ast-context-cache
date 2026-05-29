package components

import (
	"fmt"
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

func (h IndexHealth) EmbedFileLabel(path string) string {
	if path == "" {
		return ""
	}
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			if i+1 < len(path) {
				return path[i+1:]
			}
			return path
		}
	}
	return path
}

func (h IndexHealth) EmbedRecentPreview() []string {
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
