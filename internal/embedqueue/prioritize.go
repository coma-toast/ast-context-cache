package embedqueue

import (
	"log"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const queryBoostMaxWait = 2 * time.Minute

var queryBoostMu sync.Mutex

// EnsureProjectEmbeddings promotes queued embed work for projectPath to the priority
// channel and blocks until that project's backlog is drained (or queryBoostMaxWait).
// Call before hybrid/vector search so query results include fresh embeddings.
func EnsureProjectEmbeddings(projectPath string) {
	projectPath = strings.TrimSpace(projectPath)
	if projectPath == "" || !Ready() {
		return
	}
	state, _ := embedder.HealthState()
	if state == "error" {
		return
	}
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	queryBoostMu.Lock()
	defer queryBoostMu.Unlock()
	if promoted := promoteProjectToHigh(projectPath); promoted > 0 {
		log.Printf("embedqueue: query boosted %d embed job(s) for %s", promoted, projectPath)
	}
	deadline := time.Now().Add(queryBoostMaxWait)
	for time.Now().Before(deadline) {
		for {
			j, ok := stealProjectJob(projectPath)
			if !ok {
				break
			}
			run(j)
		}
		if projectInFlight(projectPath) == 0 &&
			projectPendingCount(projectPath) == 0 &&
			!channelsContainProject(projectPath) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	log.Printf("embedqueue: query boost timed out for %s", projectPath)
}

func promoteProjectToHigh(projectPath string) int {
	n := promoteChannelToHigh(lowCh, projectPath)
	n += promotePendingToHigh(projectPath)
	return n
}

func promoteChannelToHigh(ch chan job, projectPath string) int {
	if ch == nil {
		return 0
	}
	jobs := drainChannel(ch)
	n := 0
	for _, j := range jobs {
		if j.projectPath == projectPath {
			pushHigh(j)
			n++
		} else {
			ch <- j
		}
	}
	return n
}

func promotePendingToHigh(projectPath string) int {
	pendingMu.Lock()
	var promoted []job
	for k, j := range pending {
		if j.projectPath != projectPath {
			continue
		}
		promoted = append(promoted, j)
		delete(pending, k)
		delete(pendingChQueued, k)
		clearPendingDB(j)
	}
	pendingMu.Unlock()
	for _, j := range promoted {
		pushHigh(j)
	}
	return len(promoted)
}

func pushHigh(j job) {
	if highCh == nil {
		return
	}
	select {
	case highCh <- j:
	default:
		select {
		case highCh <- j:
		case lowCh <- j:
		}
	}
}

func stealProjectJob(projectPath string) (job, bool) {
	for _, ch := range []chan job{highCh, lowCh, pendingCh} {
		if ch == nil {
			continue
		}
		jobs := drainChannel(ch)
		var stolen job
		found := false
		for _, j := range jobs {
			if !found && j.projectPath == projectPath {
				stolen, found = j, true
				continue
			}
			ch <- j
		}
		if found {
			return stolen, true
		}
	}
	pendingMu.Lock()
	defer pendingMu.Unlock()
	for k, j := range pending {
		if j.projectPath == projectPath {
			delete(pending, k)
			delete(pendingChQueued, k)
			clearPendingDB(j)
			return j, true
		}
	}
	return job{}, false
}

func drainChannel(ch chan job) []job {
	var jobs []job
	for {
		select {
		case j := <-ch:
			jobs = append(jobs, j)
		default:
			return jobs
		}
	}
}

func channelsContainProject(projectPath string) bool {
	for _, ch := range []chan job{highCh, lowCh, pendingCh} {
		if ch == nil {
			continue
		}
		jobs := drainChannel(ch)
		found := false
		for _, j := range jobs {
			if j.projectPath == projectPath {
				found = true
			}
		}
		for _, j := range jobs {
			ch <- j
		}
		if found {
			return true
		}
	}
	return false
}

func projectInFlight(projectPath string) int {
	activeMu.Lock()
	defer activeMu.Unlock()
	n := 0
	for _, pp := range activeProjects {
		if pp == projectPath {
			n++
		}
	}
	return n
}

func projectPendingCount(projectPath string) int {
	pendingMu.Lock()
	defer pendingMu.Unlock()
	n := 0
	for _, j := range pending {
		if j.projectPath == projectPath {
			n++
		}
	}
	return n
}
