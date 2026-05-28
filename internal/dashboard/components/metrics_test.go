package components

import (
	"testing"
	"time"
)

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
