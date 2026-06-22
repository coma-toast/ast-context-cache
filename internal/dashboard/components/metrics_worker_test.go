package components

import "testing"

func TestWorkerStripDotCount(t *testing.T) {
	dots, ell := workerStripDotCount(10)
	if dots != 10 || ell {
		t.Fatalf("10 workers: dots=%d ellipsis=%v", dots, ell)
	}
	dots, ell = workerStripDotCount(20)
	if dots != 20 || ell {
		t.Fatalf("20 workers: dots=%d ellipsis=%v", dots, ell)
	}
	dots, ell = workerStripDotCount(25)
	if dots != 19 || !ell {
		t.Fatalf("25 workers: dots=%d ellipsis=%v", dots, ell)
	}
}

func TestWorkerStripUsesCompact(t *testing.T) {
	if workerStripUsesCompact(15) {
		t.Fatal("15 should not use compact layout")
	}
	if !workerStripUsesCompact(16) {
		t.Fatal("16 should use compact layout")
	}
}
