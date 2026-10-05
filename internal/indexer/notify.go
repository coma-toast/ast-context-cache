package indexer

import (
	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// notifyIndexCommitted runs after every committed change to projectPath's index:
// cached search candidates for any scope containing it are dropped before anyone can
// be served rankings that predate the commit, then the dashboard is told.
func notifyIndexCommitted(projectPath string) {
	cache.Candidates.ClearProject(projectPath)
	realtime.Notify(realtime.IndexCommitted)
}
