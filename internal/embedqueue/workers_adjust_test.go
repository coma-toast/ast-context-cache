package embedqueue

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestAdjustWorkersUsesTargetWhenThrottled(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	workerMu.Lock()
	workerStop = make(chan struct{}, AbsoluteMaxWorkers)
	workerCount = 4
	workerTarget = 10
	workerMu.Unlock()
	n, err := AdjustWorkers(1)
	if err != nil {
		t.Fatal(err)
	}
	if n != 11 {
		t.Fatalf("target=%d want 11", n)
	}
	if got := WorkerTarget(); got != 11 {
		t.Fatalf("WorkerTarget=%d want 11", got)
	}
}
