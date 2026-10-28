package coverage

import (
	"database/sql"
	"math"
	"time"
)

// FeatureTrend is one feature's session coverage this window and the window before it.
type FeatureTrend struct {
	Feature  string  `json:"Feature"`
	Label    string  `json:"Label"`
	Sessions int     `json:"Sessions"`
	Pct      float64 `json:"Pct"`
	PrevPct  float64 `json:"PrevPct"`
	DeltaPct float64 `json:"DeltaPct"`
}

// Digest is the coverage section of the weekly digest: the current window against the
// one immediately before it.
type Digest struct {
	WindowDays             int            `json:"WindowDays"`
	ActiveSessions         int            `json:"ActiveSessions"`
	PrevActiveSessions     int            `json:"PrevActiveSessions"`
	SearchlessSessions     int            `json:"SearchlessSessions"`
	PrevSearchlessSessions int            `json:"PrevSearchlessSessions"`
	SearchlessPct          float64        `json:"SearchlessPct"`
	PrevSearchlessPct      float64        `json:"PrevSearchlessPct"`
	Features               []FeatureTrend `json:"Features"`
}

// WeekOverWeek computes coverage for the window ending at until and the window before it.
func WeekOverWeek(conn *sql.DB, projectPath string, days int, until time.Time) (Digest, error) {
	o := Options{Days: days, ProjectPath: projectPath, Until: until}.normalize()
	cur, err := Compute(conn, o)
	if err != nil {
		return Digest{}, err
	}
	o.Until = o.Until.AddDate(0, 0, -o.Days)
	prev, err := Compute(conn, o)
	if err != nil {
		return Digest{}, err
	}
	return Trend(cur, prev), nil
}

// Trend compares two reports feature by feature.
func Trend(cur, prev Report) Digest {
	d := Digest{
		WindowDays:             cur.WindowDays,
		ActiveSessions:         cur.ActiveSessions,
		PrevActiveSessions:     prev.ActiveSessions,
		SearchlessSessions:     cur.SearchlessSessions,
		PrevSearchlessSessions: prev.SearchlessSessions,
		SearchlessPct:          cur.SearchlessPct,
		PrevSearchlessPct:      prev.SearchlessPct,
		Features:               make([]FeatureTrend, 0, len(cur.Features)),
	}
	prevPct := map[string]float64{}
	for _, f := range prev.Features {
		prevPct[f.Feature] = f.Pct
	}
	for _, f := range cur.Features {
		p := prevPct[f.Feature]
		d.Features = append(d.Features, FeatureTrend{Feature: f.Feature, Label: f.Label, Sessions: f.Sessions, Pct: f.Pct, PrevPct: p, DeltaPct: math.Round((f.Pct-p)*10) / 10})
	}
	return d
}
