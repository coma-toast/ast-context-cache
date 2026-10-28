package watcher

import (
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// watchSpace handles a watcher request for a wtg space root. The root is never
// watched or indexed as one project: that walked every checkout under the root's
// own project_path, duplicating each repo's index and skipping sibling reuse.
// Instead each checkout is linked under the root (so tools called with the root
// as project_path search all of them) and gets its own watcher, whose catch-up
// copies unchanged files from an already-indexed sibling checkout.
func watchSpace(root string, kids []string) {
	if linked := projectlinks.LinkSpaceCheckouts(root, kids); len(linked) > 0 {
		logger.Info("Watching space as its checkouts", "space", root, "checkouts", kids)
	}
	for _, k := range kids {
		if WatchRefusal(k) != "" {
			continue
		}
		EnsureWatcher(k)
	}
}
