package components

import "testing"

func TestPendingRingCapUsesPeak(t *testing.T) {
	if got := pendingRingCap(4324, 4325); got != 4325 {
		t.Fatalf("cap=%d want 4325", got)
	}
	if got := pendingRingCap(4324, 5000); got != 5000 {
		t.Fatalf("cap=%d want 5000", got)
	}
	if got := pendingRingCap(0, 0); got != 1 {
		t.Fatalf("empty cap=%d want 1", got)
	}
}

func TestPendingFillPctFromPeak(t *testing.T) {
	h := IndexHealth{EmbedPending: 4324, EmbedPendingPeak: 4325}
	pct := h.pendingFillPct()
	if pct < 99.9 || pct > 100.1 {
		t.Fatalf("fill=%.2f want ~100", pct)
	}
}
