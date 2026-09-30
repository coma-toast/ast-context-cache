package mcp

import "github.com/coma-toast/ast-context-cache/internal/watcher"

// withResourceHealth adds server descriptor usage and this project's watcher
// state to index_status. They stay in the result when the index query itself
// fails: a server out of file descriptors used to answer index_status with a
// bare null, hiding the one thing worth knowing.
func withResourceHealth(stats map[string]interface{}, err error, projectPath string) map[string]interface{} {
	out := stats
	if out == nil {
		out = map[string]interface{}{}
	}
	if err != nil {
		out["error"] = err.Error()
	}
	out["resources"] = watcher.ResourceStatus()
	if projectPath != "" {
		out["watcher"] = watcher.ProjectStatus(projectPath)
	}
	return out
}
