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
	if got := workerStripDisplayTotal(10, 4, 4); got != 10 {
		t.Fatalf("throttled target: got %d want 10", got)
	}
	if got := workerStripDisplayTotal(4, 4, 4); got != 4 {
		t.Fatalf("equal target/effective/live: got %d want 4", got)
	}
	if got := workerStripDisplayTotal(4, 4, 6); got != 6 {
		t.Fatalf("draining: got %d want 6", got)
	}
}

func TestWorkerPillClassesDraining(t *testing.T) {
	if got := workerPillClasses(4, 2, 4, 4, 5); got != "worker-pill draining" {
		t.Fatalf("draining idle pill: got %q", got)
	}
	if got := workerPillClasses(4, 5, 4, 4, 5); got != "worker-pill draining draining-busy" {
		t.Fatalf("draining busy pill: got %q", got)
	}
	if got := workerPillClasses(2, 5, 4, 4, 4); got != "worker-pill busy" {
		t.Fatalf("in-target busy pill: got %q", got)
	}
	if got := workerPillClasses(6, 2, 10, 4, 4); got != "worker-pill pending" {
		t.Fatalf("throttled pending pill: got %q", got)
	}
}

func TestWorkerControlsTitleDraining(t *testing.T) {
	got := workerControlsTitle(2, 4, 4, 5)
	want := "Workers: 4 target · 2 busy · 1 draining"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	got = workerControlsTitle(3, 10, 4, 4)
	if got != "Workers: 10 target · 4 running (WAL throttled) · 3 busy" {
		t.Fatalf("throttled title: got %q", got)
	}
}

func TestEmbedWorkersStatus(t *testing.T) {
	h := IndexHealth{EmbedWorkers: 15, EmbedWorkersEffective: 0, EmbedActive: 0}
	if got := h.EmbedWorkersStatus(); got != "0 busy" {
		t.Fatalf("status=%q want 0 busy", got)
	}
	if got := h.EmbedWorkersWalBadge(); got != "WAL 0/15" {
		t.Fatalf("badge=%q want WAL 0/15", got)
	}
}
