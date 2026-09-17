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

// UnmarkProjectCancelled clears a project's cancelled flag. Without this, a
// deleted project that's explicitly re-indexed (index_files) gets its symbols
// re-populated fine, but every embed job for it is silently discarded by
// isProjectCancelled for the rest of the process's life — embeddings never
// come back until ast-mcp restarts.
func UnmarkProjectCancelled(projectPath string) {
	cancelledMu.Lock()
	delete(cancelledProjects, projectPath)
	cancelledMu.Unlock()
}

func resetCancelledProjectsForTest() {
	cancelledMu.Lock()
	cancelledProjects = map[string]struct{}{}
	cancelledMu.Unlock()
}
