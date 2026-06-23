package components

import "testing"

func TestLoadAvgPerCoreAndLabel(t *testing.T) {
	h := IndexHealth{LoadAvg1: 23.04, LoadAvg5: 11.54, LoadAvg15: 10.61}
	cpus := loadAvgCPUs()
	got1 := loadAvgPerCore(h.LoadAvg1)
	want1 := 23.04 / cpus
	if got1 < want1-0.001 || got1 > want1+0.001 {
		t.Fatalf("per core 1m: got %v want %v (cpus=%v)", got1, want1, cpus)
	}
	util := loadAvgUtilPct(h.LoadAvg1)
	if util != got1*100 {
		t.Fatalf("util pct: got %v want %v", util, got1*100)
	}
	if loadAvgBarWidth(util) != 100 {
		t.Fatalf("bar width should cap at 100 when util=%v", util)
	}
	if h.loadAvgLevelPct() < 85 {
		t.Fatalf("level pct should be critical when overloaded, got %v", h.loadAvgLevelPct())
	}
	label := h.LoadAvgLabel()
	if label == "" || label == "23.04 · 11.54 · 10.61" {
		t.Fatalf("label should show per-core ×: %q", label)
	}
}
