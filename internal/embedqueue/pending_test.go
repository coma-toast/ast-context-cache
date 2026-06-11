package embedqueue

import "testing"

func TestPendingTracksFailedJobs(t *testing.T) {
	pendingMu.Lock()
	pending = nil
	pendingMu.Unlock()
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

func atomicStoreFailed(n int64) {
	failed = n
}

