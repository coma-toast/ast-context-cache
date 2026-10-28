package coverage

import (
	"database/sql"
	"math"
	"regexp"
	"sort"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// DefaultDays is the coverage window when none is given.
	DefaultDays = 7
	// MaxDays bounds the window a caller may ask for.
	MaxDays = 90
	// DefaultTopProjects is how many projects a report lists.
	DefaultTopProjects = 5
	// tsLayout matches the local-time prefix of queries.timestamp (RFC3339 written in local
	// time), so string comparison against it uses idx_queries_timestamp.
	tsLayout = "2006-01-02T15:04:05"
	// selectSessionToolsQuery is one row per (session, tool, project) in the window.
	selectSessionToolsQuery = `SELECT COALESCE(session_id,''), tool_name, COALESCE(project_path,''), COUNT(*),
		COALESCE(SUM(CASE WHEN COALESCE(error,'') <> '' THEN 1 ELSE 0 END),0)
		FROM queries WHERE timestamp >= ? AND timestamp < ? AND `
	projectFilterClause    = ` AND project_path = ?`
	sessionToolsGroupQuery = ` GROUP BY 1, 2, 3`
)

// anonymousSessionRe matches the hourly bucket the query log assigns calls without a
// session_id (db.extractSessionID).
var anonymousSessionRe = regexp.MustCompile(`^session-\d+$`)

// Options selects the window and scope of a report.
type Options struct {
	// Days is the window length; 0 means DefaultDays, values are clamped to MaxDays.
	Days int
	// ProjectPath limits the report to one project when set.
	ProjectPath string
	// Until ends the window (exclusive); zero means now.
	Until time.Time
	// TopProjects caps Report.TopProjects; 0 means DefaultTopProjects.
	TopProjects int
}

// FeatureUsage is one feature's session coverage.
type FeatureUsage struct {
	Feature  string  `json:"Feature"`
	Label    string  `json:"Label"`
	Sessions int     `json:"Sessions"`
	Pct      float64 `json:"Pct"`
	Calls    int     `json:"Calls"`
}

// ProjectUsage is one project's activity in the window.
type ProjectUsage struct {
	ProjectPath    string `json:"ProjectPath"`
	Sessions       int    `json:"Sessions"`
	Calls          int    `json:"Calls"`
	SearchSessions int    `json:"SearchSessions"`
}

// Report is per-session feature coverage for a window.
type Report struct {
	WindowDays  int    `json:"WindowDays"`
	Since       string `json:"Since"`
	Until       string `json:"Until"`
	ProjectPath string `json:"ProjectPath,omitempty"`
	// ActiveSessions is sessions with at least one non-internal call.
	ActiveSessions int `json:"ActiveSessions"`
	// AnonymousSessions is the subset of ActiveSessions that are hourly buckets for calls
	// made without a session_id; many of them mean agents are not passing session_id.
	AnonymousSessions  int            `json:"AnonymousSessions"`
	Calls              int            `json:"Calls"`
	ErrorCalls         int            `json:"ErrorCalls"`
	SearchlessSessions int            `json:"SearchlessSessions"`
	SearchlessPct      float64        `json:"SearchlessPct"`
	Features           []FeatureUsage `json:"Features"`
	// OtherTools counts calls to tools no feature maps (a new tool missing from Features).
	OtherTools  map[string]int `json:"OtherTools,omitempty"`
	TopProjects []ProjectUsage `json:"TopProjects"`
}

type sessionAgg struct {
	features map[string]bool
}

type projectAgg struct {
	sessions map[string]bool
	search   map[string]bool
	calls    int
}

// Compute builds a coverage report from the queries table.
func Compute(conn *sql.DB, o Options) (Report, error) {
	o = o.normalize()
	since, until := o.Until.AddDate(0, 0, -o.Days), o.Until
	r := Report{WindowDays: o.Days, Since: since.Format(time.RFC3339), Until: until.Format(time.RFC3339), ProjectPath: o.ProjectPath, Features: []FeatureUsage{}, TopProjects: []ProjectUsage{}}
	if conn == nil {
		r.Features = emptyFeatures()
		return r, nil
	}
	q := selectSessionToolsQuery + excludeInternalClause()
	args := []any{since.Format(tsLayout), until.Format(tsLayout)}
	if o.ProjectPath != "" {
		q += projectFilterClause
		args = append(args, o.ProjectPath)
	}
	rows, err := conn.Query(q+sessionToolsGroupQuery, args...)
	if err != nil {
		return r, errs.Wrap(err, "op", "coverage query", "days", o.Days, "project", o.ProjectPath)
	}
	defer rows.Close()
	sessions := map[string]*sessionAgg{}
	projects := map[string]*projectAgg{}
	featureCalls := map[string]int{}
	for rows.Next() {
		var sid, tool, project string
		var calls, errCalls int
		if err := rows.Scan(&sid, &tool, &project, &calls, &errCalls); err != nil {
			return r, errs.Wrap(err, "op", "coverage scan")
		}
		r.Calls += calls
		r.ErrorCalls += errCalls
		s := sessions[sid]
		if s == nil {
			s = &sessionAgg{features: map[string]bool{}}
			sessions[sid] = s
		}
		p := projects[project]
		if p == nil {
			p = &projectAgg{sessions: map[string]bool{}, search: map[string]bool{}}
			projects[project] = p
		}
		p.sessions[sid] = true
		p.calls += calls
		f := FeatureOf(tool)
		if f == "" {
			if r.OtherTools == nil {
				r.OtherTools = map[string]int{}
			}
			r.OtherTools[tool] += calls
			continue
		}
		s.features[f] = true
		featureCalls[f] += calls
		if f == FeatureSearch {
			p.search[sid] = true
		}
	}
	if err := rows.Err(); err != nil {
		return r, errs.Wrap(err, "op", "coverage rows")
	}
	r.ActiveSessions = len(sessions)
	featureSessions := map[string]int{}
	for sid, s := range sessions {
		if sid == "" || anonymousSessionRe.MatchString(sid) {
			r.AnonymousSessions++
		}
		for f := range s.features {
			featureSessions[f]++
		}
		if !s.features[FeatureSearch] {
			r.SearchlessSessions++
		}
	}
	r.SearchlessPct = pct(r.SearchlessSessions, r.ActiveSessions)
	for _, f := range Features {
		n := featureSessions[f.Name]
		r.Features = append(r.Features, FeatureUsage{Feature: f.Name, Label: f.Label, Sessions: n, Pct: pct(n, r.ActiveSessions), Calls: featureCalls[f.Name]})
	}
	r.TopProjects = topProjects(projects, o.TopProjects)
	return r, nil
}

func (o Options) normalize() Options {
	if o.Days <= 0 {
		o.Days = DefaultDays
	}
	o.Days = min(o.Days, MaxDays)
	if o.Until.IsZero() {
		o.Until = time.Now()
	}
	if o.TopProjects <= 0 {
		o.TopProjects = DefaultTopProjects
	}
	return o
}

func emptyFeatures() []FeatureUsage {
	out := make([]FeatureUsage, 0, len(Features))
	for _, f := range Features {
		out = append(out, FeatureUsage{Feature: f.Name, Label: f.Label})
	}
	return out
}

func topProjects(projects map[string]*projectAgg, limit int) []ProjectUsage {
	out := make([]ProjectUsage, 0, len(projects))
	for path, p := range projects {
		out = append(out, ProjectUsage{ProjectPath: path, Sessions: len(p.sessions), Calls: p.calls, SearchSessions: len(p.search)})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Sessions != out[j].Sessions {
			return out[i].Sessions > out[j].Sessions
		}
		if out[i].Calls != out[j].Calls {
			return out[i].Calls > out[j].Calls
		}
		return out[i].ProjectPath < out[j].ProjectPath
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// pct is n/total as a percentage rounded to one decimal, 0 when total is 0.
func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return math.Round(float64(n)*1000/float64(total)) / 10
}
