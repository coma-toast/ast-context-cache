package startup

import "testing"

func TestStartupPhases(t *testing.T) {
	mu.Lock()
	phase = PhaseStarting
	message = "Starting up…"
	errText = ""
	mu.Unlock()
	if !Starting() || Ready() || Failed() {
		t.Fatal("expected starting phase")
	}
	SetMessage("Loading embedder…")
	if Message() != "Loading embedder…" {
		t.Fatalf("message=%q", Message())
	}
	MarkReady()
	if !Ready() || Starting() {
		t.Fatal("expected ready phase")
	}
	if Message() != "" {
		t.Fatalf("message should clear on ready, got %q", Message())
	}
	MarkFailed("boom")
	if !Failed() || Error() != "boom" {
		t.Fatalf("failed phase err=%q", Error())
	}
	MarkReady()
	if !Ready() {
		t.Fatal("MarkReady should recover from failed")
	}
}
