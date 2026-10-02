package embedqueue

import "github.com/coma-toast/ast-context-cache/internal/realtime"

// ForgetFile drops one file from the pending retry set (memory and the batched
// embed_pending writer) after its index rows were purged, so recovery and
// FlushPending stop re-queuing it. Jobs already sitting in a channel are left
// alone — draining channels here could block behind submitters — and are
// dropped cheaply when a worker reaches them (indexer.EmbedFileSymbols sees the
// file is gone). Wired to indexer.OnFilePurged.
func ForgetFile(file, projectPath string) {
	j := job{file: file, projectPath: projectPath}
	k := jobKey(j)
	pendingMu.Lock()
	_, had := pending[k]
	delete(pending, k)
	delete(pendingChQueued, k)
	n := len(pending)
	pendingMu.Unlock()
	// Schedule the DB delete too: a queued upsert for this file must not land
	// after PurgeFile removed the row.
	clearPendingDB(j)
	trackPendingPeak(n)
	if had {
		realtime.Notify(realtime.EmbedFinished)
	}
}
