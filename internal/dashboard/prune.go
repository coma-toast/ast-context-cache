package dashboard

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/purge"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// PruneSnapshot is a point-in-time view of a database prune for the dashboard.
type PruneSnapshot struct {
	Active          bool
	Done            bool
	Phase           string
	StartedAt       time.Time
	FinishedAt      time.Time
	Error           string
	SizeBeforeBytes int64
	SizeAfterBytes  int64
	ProjectsPurged  int
	OrphanVectors   int
	QueriesPruned   int64
}

var (
	pruneMu sync.RWMutex
	prune   PruneSnapshot
)

// GetPruneSnapshot returns the current prune state for the dashboard.
func GetPruneSnapshot() PruneSnapshot {
	pruneMu.RLock()
	defer pruneMu.RUnlock()
	return prune
}

func setPrune(s PruneSnapshot) {
	pruneMu.Lock()
	prune = s
	pruneMu.Unlock()
	realtime.Notify(realtime.Settings)
}

// StartPrune reclaims disk space: it clears out data that's safe to remove (projects
// whose directory is gone, vectors with no matching symbol, query history past
// retention) and then runs VACUUM on all three databases so the freed space actually
// shrinks the files on disk rather than just becoming reusable free pages. VACUUM needs
// roughly as much free space as the current database size to build the compacted copy,
// so this can still fail with a disk-full error on a nearly-full volume; the error is
// surfaced as-is rather than worked around.
func StartPrune() (started bool, errMsg string) {
	if GetPruneSnapshot().Active {
		return false, "a prune is already in progress"
	}
	setPrune(PruneSnapshot{Active: true, Phase: "starting", StartedAt: time.Now(), SizeBeforeBytes: db.MainDBFilesSizeBytes()})
	go runPrune()
	return true, ""
}

func runPrune() {
	defer func() {
		if r := recover(); r != nil {
			pruneErrorf("prune panicked: %v", r)
		}
	}()
	snap := GetPruneSnapshot()

	snap.Phase = "sweeping deleted projects"
	setPrune(snap)
	snap.ProjectsPurged = len(purge.SweepDeletedProjects())

	snap.Phase = "removing orphaned vectors"
	setPrune(snap)
	snap.OrphanVectors = search.PurgeOrphanCodeVectors()

	snap.Phase = "pruning old query history"
	setPrune(snap)
	if n := db.RunQueryRetention(); n > 0 {
		snap.QueriesPruned = n
	}

	snap.Phase = "compacting database (VACUUM)"
	setPrune(snap)
	db.Compact()

	snap.Active = false
	snap.Done = true
	snap.Phase = "done"
	snap.FinishedAt = time.Now()
	snap.SizeAfterBytes = db.MainDBFilesSizeBytes()
	setPrune(snap)
	log.Printf("prune: reclaimed %s (%s -> %s), %d project(s) swept, %d orphan vector(s), %d old queries pruned",
		db.FormatFileSize(snap.SizeBeforeBytes-snap.SizeAfterBytes), db.FormatFileSize(snap.SizeBeforeBytes), db.FormatFileSize(snap.SizeAfterBytes),
		snap.ProjectsPurged, snap.OrphanVectors, snap.QueriesPruned)
}

func pruneErrorf(format string, args ...interface{}) {
	snap := GetPruneSnapshot()
	snap.Active = false
	snap.Done = false
	snap.Phase = "error"
	snap.Error = fmt.Sprintf(format, args...)
	snap.FinishedAt = time.Now()
	setPrune(snap)
}
