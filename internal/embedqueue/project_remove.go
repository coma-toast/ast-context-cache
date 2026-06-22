package embedqueue

import (
	"log"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// RemoveProject drops queued and pending embed work for projectPath immediately.
// In-flight jobs for the project are ignored when they finish (no re-queue).
func RemoveProject(projectPath string) (queuedRemoved, pendingRemoved int) {
	projectPath = watcher.NormalizeProjectPath(strings.TrimSpace(projectPath))
	if projectPath == "" {
		return 0, 0
	}
	markProjectCancelled(projectPath)
	drainMu.Lock()
	defer drainMu.Unlock()
	queuedRemoved = discardChannelJobs(highCh, projectPath)
	queuedRemoved += discardChannelJobs(lowCh, projectPath)
	queuedRemoved += discardChannelJobs(pendingCh, projectPath)
	pendingMu.Lock()
	for k, j := range pending {
		if j.projectPath != projectPath {
			continue
		}
		delete(pending, k)
		delete(pendingChQueued, k)
		pendingRemoved++
	}
	pendingMu.Unlock()
	purgePendingDirtyForProject(projectPath)
	if db.DB != nil {
		if res, err := db.DB.Exec(`DELETE FROM embed_pending WHERE project_path = ?`, projectPath); err != nil {
			log.Printf("embedqueue: delete embed_pending for %s: %v", projectPath, err)
		} else if n, err := res.RowsAffected(); err == nil && int(n) > pendingRemoved {
			pendingRemoved = int(n)
		}
	}
	purgeActivityForProject(projectPath)
	trackPendingPeak(PendingCount())
	if queuedRemoved > 0 || pendingRemoved > 0 {
		log.Printf("embedqueue: removed %d queued + %d pending for %s", queuedRemoved, pendingRemoved, projectPath)
		realtime.Notify(realtime.EmbedFinished)
	}
	return queuedRemoved, pendingRemoved
}

func discardChannelJobs(ch chan job, projectPath string) int {
	if ch == nil {
		return 0
	}
	jobs := drainChannel(ch)
	n := 0
	for _, j := range jobs {
		if j.projectPath == projectPath {
			unmarkPendingChQueued(j)
			n++
			continue
		}
		ch <- j
	}
	return n
}

func purgePendingDirtyForProject(projectPath string) {
	pendingDirtyMu.Lock()
	defer pendingDirtyMu.Unlock()
	for k, row := range pendingDirty {
		if row.j.projectPath == projectPath {
			delete(pendingDirty, k)
		}
	}
	for k, j := range pendingDelete {
		if j.projectPath == projectPath {
			delete(pendingDelete, k)
		}
	}
}

func purgeActivityForProject(projectPath string) {
	activeMu.Lock()
	for f, pp := range activeProjects {
		if pp == projectPath {
			delete(activeJobs, f)
			delete(activeProjects, f)
		}
	}
	activeMu.Unlock()
	recentActivityMu.Lock()
	for i := range recentActivity {
		if recentActivity[i].ProjectPath == projectPath {
			recentActivity[i] = ActivityEntry{}
		}
	}
	recentActivityMu.Unlock()
}
