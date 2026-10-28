package dashboard

import (
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// Handlers refresh the projects cache in the background, reading the db pools
// after the handler (and often the test) has returned. Watcher handlers start
// catch-up goroutines that read the pools too (sibling reuse looks up indexed
// projects), and DeleteWatcher doesn't wait for them, so stop every watcher
// before a test re-opens the pools and before its cleanup closes them.
func init() {
	dbtest.WaitFor(projectsRefreshes.Wait)
	dbtest.WaitFor(watcher.StopAll)
}
