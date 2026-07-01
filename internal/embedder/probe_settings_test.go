package embedder

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestProbeIntervalDefault(t *testing.T) {
	db.SetSetting(probeIntervalSetting, "")
	if got := ProbeInterval(); got != defaultProbeInterval {
		t.Fatalf("ProbeInterval()=%s want %s", got, defaultProbeInterval)
	}
}

func TestProbeIntervalFromSetting(t *testing.T) {
	if err := db.SetSetting(probeIntervalSetting, "45"); err != nil {
		t.Fatal(err)
	}
	defer db.SetSetting(probeIntervalSetting, "")
	if got := ProbeInterval(); got != 45*time.Second {
		t.Fatalf("ProbeInterval()=%s want 45s", got)
	}
}

func TestProbeIntervalClampsInvalid(t *testing.T) {
	if err := db.SetSetting(probeIntervalSetting, "2"); err != nil {
		t.Fatal(err)
	}
	defer db.SetSetting(probeIntervalSetting, "")
	if got := ProbeInterval(); got != defaultProbeInterval {
		t.Fatalf("below min: got %s want default %s", got, defaultProbeInterval)
	}
	if err := db.SetSetting(probeIntervalSetting, "9999"); err != nil {
		t.Fatal(err)
	}
	if got := ProbeInterval(); got != maxProbeInterval {
		t.Fatalf("above max: got %s want %s", got, maxProbeInterval)
	}
}
