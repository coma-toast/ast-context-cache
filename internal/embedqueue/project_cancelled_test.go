package embedqueue

import "testing"

// A project explicitly deleted then explicitly re-indexed (index_files) must
// resume generating embeddings — without UnmarkProjectCancelled, the process-
// lifetime cancellation flag set by RemoveProject silently discards every job
// for the re-added project until ast-mcp restarts.
func TestUnmarkProjectCancelledResumesQueuing(t *testing.T) {
	resetCancelledProjectsForTest()
	l := resetLanesForTest(t, 8)

	proj := "/proj/reindexed"
	markProjectCancelled(proj)
	if !isProjectCancelled(proj) {
		t.Fatal("project should be cancelled")
	}
	SubmitPriority("/tmp/a.go", proj, false)
	if len(l.high)+len(l.low) != 0 {
		t.Fatal("cancelled project should not enqueue")
	}

	UnmarkProjectCancelled(proj)
	if isProjectCancelled(proj) {
		t.Fatal("project should no longer be cancelled after UnmarkProjectCancelled")
	}
	SubmitPriority("/tmp/b.go", proj, false)
	if len(l.high)+len(l.low) == 0 {
		t.Fatal("re-indexed project should resume enqueuing after UnmarkProjectCancelled")
	}
}
