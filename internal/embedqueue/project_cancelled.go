package embedqueue

import "sync"

var (
	cancelledMu       sync.Mutex
	cancelledProjects = map[string]struct{}{}
)

func markProjectCancelled(projectPath string) {
	cancelledMu.Lock()
	cancelledProjects[projectPath] = struct{}{}
	cancelledMu.Unlock()
}

func isProjectCancelled(projectPath string) bool {
	cancelledMu.Lock()
	_, ok := cancelledProjects[projectPath]
	cancelledMu.Unlock()
	return ok
}

func resetCancelledProjectsForTest() {
	cancelledMu.Lock()
	cancelledProjects = map[string]struct{}{}
	cancelledMu.Unlock()
}
