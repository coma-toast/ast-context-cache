package components

import (
	"fmt"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
)

type Health struct {
	EmbedderState         string
	EmbedderError         string
	EmbedderLast          time.Duration
	EmbedBackend          string
	EmbedModel            string
	EmbedRuntime          string
	EmbedDim              int
	QueueWorkers          int
	QueueWorkersEffective int
	QueueWorkersLive      int
	QueueThroughput       int64
	QueueQueued           int
	QueuePending          int
	QueuePendingPeak      int
	QueueInFlight         int64
	QueueHighCap          int
	QueueLowCap           int
	CacheHitRatio         float64
	VectorMemMB           float64
	HeapMB                float64
	CPUPercent            float64
	TotalAllocMB          float64
	Uptime                time.Duration
	Version               string
	AbnormalPreviousRun   bool
}

func (h Health) EmbedderEmoji() string {
	switch h.EmbedderState {
	case "ready":
		return "✅"
	case "degraded":
		return "⚠️"
	case "loading":
		return "⏳"
	case "error":
		return "❌"
	default:
		return "💤"
	}
}

func (h Health) EmbedderStatus() string {
	if h.EmbedderState == "loading" {
		if msg := h.StartupMessage(); msg != "" {
			return msg
		}
		return "Starting…"
	}
	if h.EmbedderState == "error" || h.EmbedderState == "degraded" {
		return h.EmbedderErrorShort()
	}
	if h.EmbedderState == "idle" {
		return "Idle"
	}
	if h.EmbedderLast < time.Second {
		return "just now"
	}
	if h.EmbedderLast < time.Minute {
		return fmt.Sprintf("%.0fs ago", h.EmbedderLast.Seconds())
	}
	return fmt.Sprintf("%.0fm ago", h.EmbedderLast.Minutes())
}

func (h Health) CachePercent() int {
	return int(h.CacheHitRatio * 100)
}

func (h Health) FormatUptime() string {
	if h.Uptime < time.Minute {
		return fmt.Sprintf("%.0fs", h.Uptime.Seconds())
	}
	if h.Uptime < time.Hour {
		return fmt.Sprintf("%.0fm", h.Uptime.Minutes())
	}
	if h.Uptime < 24*time.Hour {
		return fmt.Sprintf("%.1fh", h.Uptime.Hours())
	}
	return fmt.Sprintf("%.0fd", h.Uptime.Hours()/24)
}

func truncateEmbedLabel(s string, max int) string {
	if max <= 3 || len(s) <= max {
		return s
	}
	return s[:max-1] + "…"
}

func (h IndexHealth) ActiveEmbedTitle() string {
	if h.EmbedEndpoint != "" {
		return fmt.Sprintf("%s · %s · %s", h.EmbedBackend, h.EmbedModel, h.EmbedEndpoint)
	}
	return fmt.Sprintf("%s · %s · %s (%d dims)", h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, h.EmbedDim)
}

func (h IndexHealth) ActiveEmbedModelShort() string {
	return truncateEmbedLabel(h.EmbedModel, 42)
}

func (h Health) ActiveEmbedModelShort() string {
	return truncateEmbedLabel(h.EmbedModel, 28)
}

func (h Health) ActiveEmbedTitle() string {
	return fmt.Sprintf("%s · %s · %s (%d dims)", h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, h.EmbedDim)
}

func (h Health) ActiveEmbedTitleFull() string {
	t := h.ActiveEmbedTitle()
	if (h.EmbedderState == "error" || h.EmbedderState == "degraded") && h.EmbedderError != "" {
		return t + " — " + h.EmbedderError
	}
	return t
}

func (h Health) ShowQueryCache() bool {
	return h.CacheHitRatio > 0
}

func (h Health) EmbedderErrorShort() string {
	return truncateEmbedLabel(h.EmbedderHumanError(), 48)
}

func (h Health) EmbedderHumanError() string {
	return embedder.HumanizeEmbedError(h.EmbedderError)
}

func (h Health) EmbedderChipLabel() string {
	switch h.EmbedderState {
	case "error":
		return "error"
	case "degraded":
		return "degraded"
	case "loading":
		return "loading"
	case "ready":
		return "ok"
	default:
		return "idle"
	}
}

func (h Health) EmbedderChipClass() string {
	switch h.EmbedderState {
	case "error":
		return "health-status-badge error"
	case "degraded":
		return "health-status-badge degraded"
	case "loading":
		return "health-status-badge loading"
	case "ready":
		return "health-status-badge ok"
	default:
		return "health-status-badge idle"
	}
}
