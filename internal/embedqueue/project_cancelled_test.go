package embedqueue

import "testing"

// A project explicitly deleted then explicitly re-indexed (index_files) must
// resume generating embeddings — without UnmarkProjectCancelled, the process-
// lifetime cancellation flag set by RemoveProject silently discards every job
// for the re-added project until ast-mcp restarts.
func TestUnmarkProjectCancelledResumesQueuing(t *testing.T) {
	resetCancelledProjectsForTest()
	highCh = make(chan job, 8)
	lowCh = make(chan job, 8)
	pendingCh = make(chan job, 8)
	pendingMu.Lock()
	pending = map[string]job{}
	pendingChQueued = map[string]struct{}{}
	pendingMu.Unlock()

	proj := "/proj/reindexed"
	markProjectCancelled(proj)
	if !isProjectCancelled(proj) {
		t.Fatal("project should be cancelled")
	}
	SubmitPriority("/tmp/a.go", proj, false)
	if len(highCh)+len(lowCh) != 0 {
		t.Fatal("cancelled project should not enqueue")
	}

	UnmarkProjectCancelled(proj)
	if isProjectCancelled(proj) {
		t.Fatal("project should no longer be cancelled after UnmarkProjectCancelled")
	}
	SubmitPriority("/tmp/b.go", proj, false)
	if len(highCh)+len(lowCh) == 0 {
		t.Fatal("re-indexed project should resume enqueuing after UnmarkProjectCancelled")
	}
}
