package dashboard

import (
	"encoding/json"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/coverage"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

// coverageMetricsTTL caches the all-projects report the /metrics gauges read, so a scrape
// runs the coverage query at most once per TTL rather than once per gauge.
const coverageMetricsTTL = time.Minute

var coverageMetrics struct {
	sync.Mutex
	at     time.Time
	report coverage.Report
}

func handleDashboardFeatureCoverageJSON(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	days, _ := strconv.Atoi(q.Get("days"))
	rep, err := coverage.Compute(db.DB, coverage.Options{Days: days, ProjectPath: q.Get("project_id")})
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		logger.Warn("Feature coverage query failed", "error", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	json.NewEncoder(w).Encode(rep)
}

// weeklyCoverage is the digest's week-over-week coverage section; nil when the query fails.
func weeklyCoverage(projectID string, days int) *coverage.Digest {
	d, err := coverage.WeekOverWeek(db.DB, projectID, days, time.Now())
	if err != nil {
		logger.Warn("Weekly coverage query failed", "error", err, "project", projectID)
		return nil
	}
	return &d
}

// cachedCoverageReport returns the 7-day all-projects report, refreshed every coverageMetricsTTL.
func cachedCoverageReport() coverage.Report {
	coverageMetrics.Lock()
	defer coverageMetrics.Unlock()
	if !coverageMetrics.at.IsZero() && time.Since(coverageMetrics.at) < coverageMetricsTTL {
		return coverageMetrics.report
	}
	rep, err := coverage.Compute(db.DB, coverage.Options{})
	if err != nil {
		logger.Warn("Feature coverage metrics query failed", "error", err)
		return coverageMetrics.report
	}
	coverageMetrics.at, coverageMetrics.report = time.Now(), rep
	return rep
}

// featureCoverageRatio is a feature's 7-day session coverage as 0..1.
func featureCoverageRatio(feature string) float64 {
	rep := cachedCoverageReport()
	if rep.ActiveSessions == 0 {
		return 0
	}
	for _, f := range rep.Features {
		if f.Feature == feature {
			return float64(f.Sessions) / float64(rep.ActiveSessions)
		}
	}
	return 0
}
