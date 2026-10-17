package startup

import (
	"os"
	"testing"
)

func TestStartupDataDirUnavailablePhase(t *testing.T) {
	mu.Lock()
	phase = PhaseStarting
	message = "Starting up…"
	errText = ""
	mu.Unlock()

	// Test that DataDirUnavailable returns false initially
	if DataDirUnavailable() {
		t.Fatal("expected DataDirUnavailable to be false initially")
	}

	// Simulate marking data dir unavailable
	MarkDataDirUnavailable("test error")

	// Test that DataDirUnavailable returns true after marking
	if !DataDirUnavailable() {
		t.Fatal("expected DataDirUnavailable to be true after MarkDataDirUnavailable")
	}

	// Test that Error() returns the error message
	if Error() != "test error" {
		t.Fatalf("expected Error() to return 'test error', got '%s'", Error())
	}

	// Test that Message() returns "Data directory unavailable"
	if Message() != "Data directory unavailable" {
		t.Fatalf("expected Message() to return 'Data directory unavailable', got '%s'", Message())
	}

	// Clean up and reset
	os.Remove("/tmp/test-data-dir-override") // just a cleanup
}
