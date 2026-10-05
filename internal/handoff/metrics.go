package handoff

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	// repeatRatioWindow is the window astcache_handoff_repeat_search_ratio covers: children
	// active within it count (OB-1, OB-5).
	repeatRatioWindow = 24 * time.Hour

	countLiveTreesQuery    = `SELECT COUNT(*) FROM handoff_trees WHERE last_access_at >= ?`
	countOpenChildrenQuery = `SELECT COUNT(*) FROM handoff_children WHERE status = '` + string(StatusOpen) + `'`
	sumRecentSearchesQuery = `SELECT COALESCE(SUM(search_calls), 0), COALESCE(SUM(repeat_calls), 0)
		FROM handoff_children WHERE last_activity_at >= ?`
)

var (
	handoffsCreated = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "astcache_handoffs_created_total",
		Help: "Handoffs created by a parent session.",
	})
	childrenOpened = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "astcache_handoff_children_opened_total",
		Help: "Child sessions minted by open_handoff.",
	})
	childrenResumed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "astcache_handoff_children_resumed_total",
		Help: "open_handoff calls that resumed an existing child session (paging a digest is not counted).",
	})
	childrenCompleted = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "astcache_handoff_children_completed_total",
		Help: "Child completions, by status (done, partial, failed).",
	}, []string{"status"})
	childrenAbandoned = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "astcache_handoff_children_abandoned_total",
		Help: "Open children marked abandoned after the inactivity window.",
	})
	treesExpired = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "astcache_handoff_trees_expired_total",
		Help: "Handoff trees flushed by the expiry sweeper after the TTL.",
	})
	childSearches = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "astcache_handoff_child_searches_total",
		Help: "Search calls by handoff children, by whether they repeated the parent's exploration (OB-1).",
	}, []string{"repeat"})
	treeTokens = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "astcache_handoff_tree_tokens",
		Help:    "A tree's token usage after a handoff snapshot or a child result is charged to it.",
		Buckets: prometheus.ExponentialBuckets(500, 2, 10),
	})
	claimWaitSeconds = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "astcache_handoff_claim_wait_seconds",
		Help:    "How long a queued claim waited before it was granted.",
		Buckets: []float64{1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600},
	})
	openTreesGauge = prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "astcache_handoff_open_trees",
		Help: "Handoff trees accessed within the TTL.",
	}, liveTrees)
	openChildrenGauge = prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "astcache_handoff_open_children",
		Help: "Handoff children with status open.",
	}, openChildren)
	repeatRatioGauge = prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "astcache_handoff_repeat_search_ratio",
		Help: "Repeat searches over search calls by children active in the last 24h (OB-1); 0 with no searches.",
	}, RepeatSearchRatio)
)

// Collectors returns the handoff Prometheus collectors for the dashboard's /metrics (OB-5).
// The gauges read context.db when scraped and report 0 before it is open. Every label value of
// the vectors is created up front, so each series is scraped (as 0) before its first event.
func Collectors() []prometheus.Collector {
	for _, st := range []Status{StatusDone, StatusPartial, StatusFailed} {
		childrenCompleted.WithLabelValues(string(st))
	}
	childSearches.WithLabelValues("true")
	childSearches.WithLabelValues("false")
	return []prometheus.Collector{
		handoffsCreated, childrenOpened, childrenResumed, childrenCompleted, childrenAbandoned, treesExpired, childSearches,
		openTreesGauge, openChildrenGauge, repeatRatioGauge, treeTokens, claimWaitSeconds,
	}
}

// RepeatSearchRatio is repeat_calls over search_calls summed across children active in the
// last 24 hours, or 0 when they made no searches.
func RepeatSearchRatio() float64 {
	if db.ContextDB == nil {
		return 0
	}
	var searches, repeats int
	cutoff := sqlTime(nowFunc().Add(-repeatRatioWindow))
	if err := db.ContextDB.QueryRow(sumRecentSearchesQuery, cutoff).Scan(&searches, &repeats); err != nil {
		return 0
	}
	return repeatRate(searches, repeats)
}

// repeatRate is repeats over searches (OB-1), 0 with no searches.
func repeatRate(searches, repeats int) float64 {
	if searches == 0 {
		return 0
	}
	return float64(repeats) / float64(searches)
}

func liveTrees() float64 {
	return countRows(countLiveTreesQuery, sqlTime(nowFunc().Add(-LoadLimits().TTL())))
}

func openChildren() float64 {
	return countRows(countOpenChildrenQuery)
}

func countRows(q string, args ...any) float64 {
	if db.ContextDB == nil {
		return 0
	}
	var n int
	if err := db.ContextDB.QueryRow(q, args...).Scan(&n); err != nil {
		return 0
	}
	return float64(n)
}

// observeClaimWaitSeconds is the default onClaimWait.
func observeClaimWaitSeconds(d time.Duration) {
	claimWaitSeconds.Observe(d.Seconds())
}

// countChildSearch counts one child search for astcache_handoff_child_searches_total.
func countChildSearch(repeat bool) {
	childSearches.WithLabelValues(strconv.FormatBool(repeat)).Inc()
}

// notifyDashboard tells the dashboard the handoff trees changed; realtime debounces bursts.
func notifyDashboard() {
	realtime.Notify(realtime.Handoffs)
}
