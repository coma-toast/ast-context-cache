package embedqueue

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// RemoveProjectFilesUnder drops queued/pending embed work for files under child owned by parent project_path.
func RemoveProjectFilesUnder(parent, child string) (queuedRemoved, pendingRemoved int) {
	parent = watcher.NormalizeProjectPath(strings.TrimSpace(parent))
	child = watcher.NormalizeProjectPath(strings.TrimSpace(child))
	if parent == "" || child == "" {
		return 0, 0
	}
	prefix := child
	if !strings.HasSuffix(prefix, string('/')) {
		prefix += "/"
	}
	match := func(file, projectPath string) bool {
		if projectPath != parent {
			return false
		}
		file = watcher.NormalizeProjectPath(file)
		return file == child || strings.HasPrefix(file, prefix)
	}
	drainMu.Lock()
	defer drainMu.Unlock()
	queuedRemoved = discardChannelJobsMatch(highCh, match)
	queuedRemoved += discardChannelJobsMatch(lowCh, match)
	queuedRemoved += discardChannelJobsMatch(pendingCh, match)
	pendingMu.Lock()
	for k, j := range pending {
		if !match(j.file, j.projectPath) {
			continue
		}
		delete(pending, k)
		delete(pendingChQueued, k)
		pendingRemoved++
	}
	pendingMu.Unlock()
	purgePendingDirtyMatch(match)
	activeMu.Lock()
	for f, pp := range activeProjects {
		if match(f, pp) {
			delete(activeJobs, f)
			delete(activeProjects, f)
		}
	}
	activeMu.Unlock()
	return queuedRemoved, pendingRemoved
}

func discardChannelJobsMatch(ch chan job, match func(file, projectPath string) bool) int {
	if ch == nil {
		return 0
	}
	jobs := drainChannel(ch)
	n := 0
	for _, j := range jobs {
		if match(j.file, j.projectPath) {
			unmarkPendingChQueued(j)
			n++
			continue
		}
		ch <- j
	}
	return n
}

func purgePendingDirtyMatch(match func(file, projectPath string) bool) {
	pendingDirtyMu.Lock()
	defer pendingDirtyMu.Unlock()
	for k, row := range pendingDirty {
		if match(row.j.file, row.j.projectPath) {
			delete(pendingDirty, k)
		}
	}
	for k, j := range pendingDelete {
		if match(j.file, j.projectPath) {
			delete(pendingDelete, k)
		}
	}
}
