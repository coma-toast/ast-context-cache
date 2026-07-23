package dashboard

import (
	"log"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	promOnce sync.Once

	mcpToolCalls = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "astcache_mcp_tool_calls_total",
		Help: "MCP / indexer tool calls flushed to the query log (by tool_name).",
	}, []string{"tool"})

	mcpToolDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "astcache_mcp_tool_duration_seconds",
		Help:    "Wall duration of MCP / indexer tool calls from the query log flush hook.",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
	}, []string{"tool"})
)

// registerPrometheusMetrics registers GaugeFuncs and counters for /metrics (once).
func registerPrometheusMetrics() {
	promOnce.Do(func() {
		mustRegister(
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_up",
				Help: "1 while the dashboard process is serving /metrics.",
			}, func() float64 { return 1 }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_pending",
				Help: "Embed jobs waiting for retry (failed / drained).",
			}, func() float64 { return float64(embedqueue.Snapshot().Pending) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_queued",
				Help: "Embed jobs currently in high+low channels.",
			}, func() float64 { return float64(embedqueue.Snapshot().Queued) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_in_flight",
				Help: "Embed jobs currently running.",
			}, func() float64 { return float64(embedqueue.Snapshot().InFlight) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_workers_target",
				Help: "Configured primary embed worker target.",
			}, func() float64 { return float64(embedqueue.Snapshot().Workers) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_workers_live",
				Help: "Live primary embed worker goroutines.",
			}, func() float64 { return float64(embedqueue.Snapshot().WorkersLive) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_aux_workers_target",
				Help: "Configured aux embed worker target.",
			}, func() float64 { return float64(embedqueue.Snapshot().AuxWorkers) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embed_aux_workers_live",
				Help: "Live aux embed worker goroutines.",
			}, func() float64 { return float64(embedqueue.Snapshot().AuxWorkersLive) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_embedder_state",
				Help: "Embedder health as numeric: idle=0 loading=1 ready=2 degraded=3 error=4 unknown=-1.",
			}, embedderStateNumeric),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_index_wal_bytes",
				Help: "On-disk size of index.db WAL.",
			}, func() float64 { return float64(db.IndexWalBytes()) }),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{
				Name: "astcache_tokens_saved_today",
				Help: "Sum of tokens_saved for today (local calendar day) from the query log.",
			}, tokensSavedToday),
			mcpToolCalls,
			mcpToolDuration,
		)
		log.Printf("dashboard: prometheus metrics registered at /metrics")
	})
}

func mustRegister(cs ...prometheus.Collector) {
	for _, c := range cs {
		if err := prometheus.Register(c); err != nil {
			if _, ok := err.(prometheus.AlreadyRegisteredError); ok {
				continue
			}
			log.Printf("dashboard: prometheus register: %v", err)
		}
	}
}

func embedderStateNumeric() float64 {
	state, _ := embedder.HealthState()
	switch state {
	case "idle":
		return 0
	case "loading":
		return 1
	case "ready":
		return 2
	case "degraded":
		return 3
	case "error":
		return 4
	default:
		return -1
	}
}

func tokensSavedToday() float64 {
	if db.DB == nil {
		return 0
	}
	todayStart := time.Now().Format("2006-01-02") + "T00:00:00"
	tomorrowStart := time.Now().AddDate(0, 0, 1).Format("2006-01-02") + "T00:00:00"
	var n int
	err := db.DB.QueryRow(
		"SELECT "+tokensSavedSum+" FROM queries WHERE timestamp >= ? AND timestamp < ?",
		todayStart, tomorrowStart,
	).Scan(&n)
	if err != nil {
		return 0
	}
	return float64(n)
}

// observeQueryLogMetrics increments tool-call counters/histograms from a flush batch.
func observeQueryLogMetrics(rows []db.QueryLogSnapshot) {
	for _, r := range rows {
		tool := r.ToolName
		if tool == "" {
			tool = "unknown"
		}
		mcpToolCalls.WithLabelValues(tool).Inc()
		if r.DurationMs > 0 {
			mcpToolDuration.WithLabelValues(tool).Observe(r.DurationMs / 1000.0)
		}
	}
}
