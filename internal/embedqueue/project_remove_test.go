package embedqueue

import "testing"

func TestRemoveProject_clearsQueuedAndPending(t *testing.T) {
	resetCancelledProjectsForTest()
	l := resetLanesForTest(t, 8)
	pendingMu.Lock()
	pending = map[string]job{
		jobKey(job{file: "/tmp/p.go", projectPath: "/proj/del"}):     {file: "/tmp/p.go", projectPath: "/proj/del"},
		jobKey(job{file: "/tmp/keep.go", projectPath: "/proj/keep"}): {file: "/tmp/keep.go", projectPath: "/proj/keep"},
	}
	pendingChQueued = map[string]struct{}{
		jobKey(job{file: "/tmp/p.go", projectPath: "/proj/del"}): {},
	}
	pendingMu.Unlock()
	proj := "/proj/del"
	other := "/proj/keep"
	l.low <- job{file: "/tmp/a.go", projectPath: proj}
	l.low <- job{file: "/tmp/b.go", projectPath: other}
	l.high <- job{file: "/tmp/c.go", projectPath: proj}
	l.pending <- job{file: "/tmp/d.go", projectPath: proj}
	q, p := RemoveProject(proj)
	if q != 3 {
		t.Fatalf("queued removed=%d want 3", q)
	}
	if p != 1 {
		t.Fatalf("pending removed=%d want 1", p)
	}
	if len(l.low) != 1 {
		t.Fatalf("low used=%d want 1", len(l.low))
	}
	if PendingCount() != 1 {
		t.Fatalf("pending left=%d want 1", PendingCount())
	}
	if isProjectCancelled(other) {
		t.Fatal("other project should not be cancelled")
	}
	if markPendingIfNew(job{file: "/tmp/new.go", projectPath: proj}, pendingReasonFailed) {
		t.Fatal("cancelled project should not accept new pending")
	}
	SubmitPriority("/tmp/x.go", proj, false)
	if len(l.low) != 1 {
		t.Fatal("cancelled project should not enqueue")
	}
}

func TestRemoveProject_blocksPendingAfterCancel(t *testing.T) {
	resetCancelledProjectsForTest()
	pendingMu.Lock()
	pending = map[string]job{}
	pendingChQueued = map[string]struct{}{}
	pendingMu.Unlock()
	proj := "/proj/gone"
	markProjectCancelled(proj)
	if markPendingIfNew(job{file: "/tmp/z.go", projectPath: proj}, pendingReasonFailed) {
		t.Fatal("should not mark pending for cancelled project")
	}
}
