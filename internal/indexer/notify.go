package indexer

import "github.com/coma-toast/ast-context-cache/internal/realtime"

func notifyIndexCommitted() {
	realtime.Notify(realtime.IndexCommitted)
}
