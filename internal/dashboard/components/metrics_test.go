package components

import (
	"testing"
	"time"
)

func TestTodayMeterFill(t *testing.T) {
	below := todayMeterFill(50, 3000) // avg 100, today 50
	if below.OverlapPct != 50 || below.DayOnlyPct != 0 || below.AvgOnlyPct != 50 {
		t.Fatalf("below avg: %+v", below)
	}
	at := todayMeterFill(100, 3000)
	if at.OverlapPct != 100 || at.DayOnlyPct != 0 || at.AvgOnlyPct != 0 {
		t.Fatalf("at avg: %+v", at)
	}
	above := todayMeterFill(150, 3000) // avg 100, today 150
	if above.OverlapPct < 66.6 || above.OverlapPct > 66.7 {
		t.Fatalf("overlap pct: got %.2f want ~66.7", above.OverlapPct)
	}
	if above.DayOnlyPct < 33.3 || above.DayOnlyPct > 33.4 {
		t.Fatalf("day only pct: got %.2f want ~33.3", above.DayOnlyPct)
	}
	if above.AvgOnlyPct != 0 {
		t.Fatalf("avg only pct: got %.2f want 0", above.AvgOnlyPct)
	}
	if above.GaugePct != 150 {
		t.Fatalf("gauge pct: got %.0f want 150", above.GaugePct)
	}
}

func TestFormatDocSourceAge(t *testing.T) {
	age, stale := FormatDocSourceAge("", 7*24*time.Hour)
	if age != "never" || !stale {
		t.Fatalf("empty: age=%q stale=%v", age, stale)
	}
	recent := time.Now().Add(-2 * time.Hour).Format(time.RFC3339)
	age, stale = FormatDocSourceAge(recent, 7*24*time.Hour)
	if stale || age != "2h" {
		t.Fatalf("recent: age=%q stale=%v", age, stale)
	}
	old := time.Now().Add(-8 * 24 * time.Hour).Format(time.RFC3339)
	age, stale = FormatDocSourceAge(old, 7*24*time.Hour)
	if !stale {
		t.Fatal("expected stale")
	}
	if age != "8d" {
		t.Fatalf("age=%q", age)
	}
}
