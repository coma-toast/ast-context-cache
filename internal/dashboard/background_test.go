package dashboard

import "github.com/coma-toast/ast-context-cache/internal/db/dbtest"

// Handlers refresh the projects cache in the background, reading the db pools
// after the handler (and often the test) has returned.
func init() { dbtest.WaitFor(projectsRefreshes.Wait) }
