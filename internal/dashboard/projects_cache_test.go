package dashboard

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
)

// Regression test for the "delete/pin/rename briefly wipes the whole Projects
// list" bug: invalidateProjectsCache used to nil out projectsCache, so the very
// next loadProjectsForPage call (from the frontend's post-action onRefresh) hit
// the empty-cache branch and returned Projects: nil, ProjectsLoading: true — with
// nothing on the client ever retrying. That collapsed pagination and reflowed
// every row for however long the background rebuild took, which is what let a
// stray click land on the wrong project mid-action.
func TestInvalidateProjectsCacheKeepsServingStaleSnapshot(t *testing.T) {
	projectsCacheMu.Lock()
	projectsCache = []components.Project{{Path: "/tmp/proj-a", Label: "proj-a"}}
	projectsCacheAt = time.Now()
	projectsCacheMu.Unlock()

	invalidateProjectsCache()

	ps, loading := loadProjectsForPage()
	if loading {
		t.Fatalf("loadProjectsForPage reported loading right after invalidate; want it to keep serving the stale snapshot")
	}
	if len(ps) != 1 || ps[0].Path != "/tmp/proj-a" {
		t.Fatalf("expected stale snapshot to still be served, got %+v", ps)
	}

	// Let the background refresh invalidate kicked off finish before the next test runs.
	time.Sleep(50 * time.Millisecond)
}
