package embedqueue

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestSetWorkerCountValidation(t *testing.T) {
	if _, err := SetWorkerCount(-1); err == nil {
		t.Fatal("expected error for negative workers")
	}
	if _, err := SetWorkerCount(MaxWorkers() + 1); err == nil {
		t.Fatal("expected error above max workers")
	}
}

func TestMaxWorkersSetting(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if err := db.SetSetting("embed_worker_max", "25"); err != nil {
		t.Fatal(err)
	}
	if got := MaxWorkers(); got != 25 {
		t.Fatalf("MaxWorkers() = %d, want 25", got)
	}
	if err := db.SetSetting("embed_worker_max", "999"); err != nil {
		t.Fatal(err)
	}
	if got := MaxWorkers(); got != AbsoluteMaxWorkers {
		t.Fatalf("MaxWorkers() = %d, want cap %d", got, AbsoluteMaxWorkers)
	}
}
