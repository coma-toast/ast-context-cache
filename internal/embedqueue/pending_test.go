package embedqueue

import "testing"

func TestPendingTracksFailedJobs(t *testing.T) {
	pendingMu.Lock()
	pending = nil
	pendingMu.Unlock()
	pendingPeak.Store(0)
	atomicStoreFailed(0)

	markPendingIfNew(job{file: "/tmp/a.go", projectPath: "/proj"}, pendingReasonFailed)
	if PendingCount() != 1 {
		t.Fatalf("pending=%d", PendingCount())
	}
	clearPending(job{file: "/tmp/a.go", projectPath: "/proj"})
	if PendingCount() != 0 {
		t.Fatalf("pending after clear=%d", PendingCount())
	}
}

func TestPendingPeakSinceZero(t *testing.T) {
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	pendingPeak.Store(0)
	for i := 0; i < 5; i++ {
		markPendingIfNew(job{file: "/tmp/f" + string(rune('a'+i)) + ".go", projectPath: "/proj"}, pendingReasonFailed)
	}
	if PendingPeak() != 5 {
		t.Fatalf("peak=%d want 5", PendingPeak())
	}
	clearPending(job{file: "/tmp/a.go", projectPath: "/proj"})
	if PendingPeak() != 5 {
		t.Fatalf("peak after one clear=%d want 5", PendingPeak())
	}
	pendingMu.Lock()
	pending = map[string]job{}
	pendingMu.Unlock()
	trackPendingPeak(0)
	if PendingPeak() != 0 {
		t.Fatalf("peak after zero=%d", PendingPeak())
	}
}

func atomicStoreFailed(n int64) {
	failed = n
}

