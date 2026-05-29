package components

import (
	"testing"
	"time"
)

func TestTodayMeterFill(t *testing.T) {
	below := todayMeterFill(50, 3000) // avg 100, today 50
	if below.AboveAvg || below.WidthPct != 50 || below.GaugePct != 50 {
		t.Fatalf("below avg: %+v", below)
	}
	at := todayMeterFill(100, 3000)
	if at.AboveAvg || at.WidthPct != 100 {
		t.Fatalf("at avg: %+v", at)
	}
	above := todayMeterFill(150, 3000) // avg 100, today 150
	if !above.AboveAvg || above.WidthPct != 100 {
		t.Fatalf("above avg width: %+v", above)
	}
	if above.AvgPct < 66.6 || above.AvgPct > 66.7 {
		t.Fatalf("avg pct: got %.2f want ~66.7", above.AvgPct)
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
