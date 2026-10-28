package coverage

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

var fixtureUntil = time.Date(2026, 10, 28, 12, 0, 0, 0, time.Local)

// seedFixture writes a small query log: four sessions this week, two the week before.
func seedFixture(t *testing.T) {
	t.Helper()
	dbtest.Init(t)
	rows := []struct {
		ago                      time.Duration
		tool, sid, project, errM string
	}{
		{time.Hour, "get_context_capsule", "s1", "/a", ""},
		{2 * time.Hour, "store_context", "s1", "/a", ""},
		{3 * time.Hour, "recall_memory", "s1", "/a", ""},
		{3 * time.Hour, "file_watcher", "s1", "/a", ""},
		{24 * time.Hour, "recall_memory", "s2", "/a", ""},
		{25 * time.Hour, "list_context", "s2", "/a", ""},
		{48 * time.Hour, "fetch_doc", "s3", "/b", ""},
		{49 * time.Hour, "handoff", "s3", "/b", ""},
		{50 * time.Hour, "mystery_tool", "s3", "/b", ""},
		{72 * time.Hour, "index_status", "session-498090", "/b", "boom"},
		{10 * 24 * time.Hour, "get_impact_graph", "s4", "/a", ""},
		{10 * 24 * time.Hour, "get_context_capsule", "s4", "/a", ""},
		{11 * 24 * time.Hour, "recall_memory", "s5", "/b", ""},
		{20 * 24 * time.Hour, "recall_memory", "s6", "/b", ""},
		{-time.Hour, "recall_memory", "future", "/b", ""},
	}
	for _, r := range rows {
		ts := fixtureUntil.Add(-r.ago).Format(time.RFC3339)
		if _, err := db.DB.Exec(`INSERT INTO queries (timestamp, tool_name, session_id, project_path, error) VALUES (?, ?, ?, ?, ?)`, ts, r.tool, r.sid, r.project, r.errM); err != nil {
			t.Fatal(err)
		}
	}
}

func featurePct(r Report) map[string]float64 {
	out := map[string]float64{}
	for _, f := range r.Features {
		out[f.Feature] = f.Pct
	}
	return out
}

func TestCompute(t *testing.T) {
	seedFixture(t)
	tests := []struct {
		name                                      string
		opts                                      Options
		active, anon, calls, errCalls, searchless int
		searchlessPct                             float64
		pcts                                      map[string]float64
		other                                     int
		topProject                                string
	}{
		{
			name: "all projects", opts: Options{Until: fixtureUntil},
			active: 4, anon: 1, calls: 9, errCalls: 1, searchless: 3, searchlessPct: 75,
			pcts:  map[string]float64{"search": 25, "virtual_context": 50, "memory": 50, "docs": 25, "handoff": 25, "index": 25, "impact": 0},
			other: 1, topProject: "/a",
		},
		{
			name: "project filter", opts: Options{Until: fixtureUntil, ProjectPath: "/a"},
			active: 2, calls: 5, searchless: 1, searchlessPct: 50,
			pcts:       map[string]float64{"search": 50, "virtual_context": 100, "memory": 100, "docs": 0},
			topProject: "/a",
		},
		{
			name: "previous week", opts: Options{Until: fixtureUntil.AddDate(0, 0, -7)},
			active: 2, calls: 3, searchless: 1, searchlessPct: 50,
			pcts:       map[string]float64{"search": 50, "impact": 50, "memory": 50},
			topProject: "/a",
		},
		{
			name: "empty window", opts: Options{Until: fixtureUntil.AddDate(-1, 0, 0)},
			pcts: map[string]float64{"search": 0},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r, err := Compute(db.DB, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			if r.WindowDays != DefaultDays || len(r.Features) != len(Features) {
				t.Fatalf("WindowDays=%d features=%d", r.WindowDays, len(r.Features))
			}
			if r.ActiveSessions != tc.active || r.AnonymousSessions != tc.anon || r.Calls != tc.calls || r.ErrorCalls != tc.errCalls {
				t.Fatalf("active=%d anon=%d calls=%d err=%d", r.ActiveSessions, r.AnonymousSessions, r.Calls, r.ErrorCalls)
			}
			if r.SearchlessSessions != tc.searchless || r.SearchlessPct != tc.searchlessPct {
				t.Fatalf("searchless=%d (%v%%)", r.SearchlessSessions, r.SearchlessPct)
			}
			got := featurePct(r)
			for f, want := range tc.pcts {
				if got[f] != want {
					t.Errorf("%s pct=%v want %v", f, got[f], want)
				}
			}
			if r.OtherTools["mystery_tool"] != tc.other {
				t.Errorf("OtherTools=%v", r.OtherTools)
			}
			if tc.topProject != "" && (len(r.TopProjects) == 0 || r.TopProjects[0].ProjectPath != tc.topProject) {
				t.Errorf("TopProjects=%+v", r.TopProjects)
			}
		})
	}
}

func TestWeekOverWeek(t *testing.T) {
	seedFixture(t)
	d, err := WeekOverWeek(db.DB, "", 7, fixtureUntil)
	if err != nil {
		t.Fatal(err)
	}
	if d.ActiveSessions != 4 || d.PrevActiveSessions != 2 || d.SearchlessSessions != 3 || d.PrevSearchlessSessions != 1 {
		t.Fatalf("digest=%+v", d)
	}
	want := map[string][3]float64{"search": {25, 50, -25}, "memory": {50, 50, 0}, "impact": {0, 50, -50}, "docs": {25, 0, 25}}
	for _, f := range d.Features {
		w, ok := want[f.Feature]
		if !ok {
			continue
		}
		if f.Pct != w[0] || f.PrevPct != w[1] || f.DeltaPct != w[2] {
			t.Errorf("%s = %v/%v/%v want %v", f.Feature, f.Pct, f.PrevPct, f.DeltaPct, w)
		}
	}
}

func TestComputeNilDB(t *testing.T) {
	r, err := Compute(nil, Options{Days: 500})
	if err != nil {
		t.Fatal(err)
	}
	if r.WindowDays != MaxDays || len(r.Features) != len(Features) || r.ActiveSessions != 0 {
		t.Fatalf("report=%+v", r)
	}
}

func TestFeatureMap(t *testing.T) {
	tests := []struct{ tool, feature string }{
		{"get_context_capsule", FeatureSearch},
		{"diff_impact", "impact"},
		{"edit_context", "virtual_context"},
		{"recall_memory", "memory"},
		{"update_doc_source", "docs"},
		{"execute_code", "code_mode"},
		{"scratchpad", "handoff"},
		{"apply_context_fn", "context_fn"},
		{"index_status", "index"},
		{"file_watcher", ""},
		{"nope", ""},
	}
	for _, tc := range tests {
		if got := FeatureOf(tc.tool); got != tc.feature {
			t.Errorf("FeatureOf(%s)=%q want %q", tc.tool, got, tc.feature)
		}
	}
	if !IsInternal("file_watcher") || IsInternal("handoff") {
		t.Fatal("IsInternal")
	}
	if len(ToolsFor(FeatureSearch)) == 0 || ToolsFor("nope") != nil {
		t.Fatal("ToolsFor")
	}
	seen := map[string]string{}
	for _, f := range Features {
		for _, tool := range f.Tools {
			if prev, ok := seen[tool]; ok {
				t.Errorf("%s in both %s and %s", tool, prev, f.Name)
			}
			seen[tool] = f.Name
		}
	}
}
