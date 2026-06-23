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

func TestWorkerStripDisplayTotal(t *testing.T) {
	if got := workerStripDisplayTotal(4, 4); got != 4 {
		t.Fatalf("equal total/live: got %d want 4", got)
	}
	if got := workerStripDisplayTotal(4, 6); got != 6 {
		t.Fatalf("draining: got %d want 6", got)
	}
}

func TestWorkerPillClassesDraining(t *testing.T) {
	if got := workerPillClasses(4, 2, 4, 5); got != "worker-pill draining" {
		t.Fatalf("draining idle pill: got %q", got)
	}
	if got := workerPillClasses(4, 5, 4, 5); got != "worker-pill draining draining-busy" {
		t.Fatalf("draining busy pill: got %q", got)
	}
	if got := workerPillClasses(2, 5, 4, 4); got != "worker-pill busy" {
		t.Fatalf("in-target busy pill: got %q", got)
	}
}

func TestWorkerControlsTitleDraining(t *testing.T) {
	got := workerControlsTitle(2, 4, 5)
	want := "Workers: 4 target · 2 busy · 1 draining"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
