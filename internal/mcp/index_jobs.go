package mcp

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// Directory index_files calls run as background jobs: a large tree takes longer than
// MCP clients wait (~30-40s), so the call waits at most indexJobSyncWait and otherwise
// returns a job the agent polls via index_status.
const (
	indexJobQueued    = "queued"
	indexJobRunning   = "running"
	indexJobCompleted = "completed"
	indexJobFailed    = "failed"

	// maxFinishedIndexJobs bounds how many finished jobs are kept for polling.
	maxFinishedIndexJobs = 20
	// maxStatusIndexJobs caps how many jobs index_status lists per project.
	maxStatusIndexJobs = 5
)

var (
	// indexJobSyncWait is how long index_files waits for a directory job before
	// returning its queued/running status. Small directories finish inside it and keep
	// the old synchronous {"indexed": n} answer.
	indexJobSyncWait = 2 * time.Second
	// runIndexDirectory is swapped in tests to control job timing.
	runIndexDirectory = indexer.IndexDirectoryProgress
	// afterIndexDirectory runs once a directory job succeeds (swapped in tests).
	afterIndexDirectory = defaultAfterIndexDirectory

	indexJobsMu  sync.Mutex
	indexJobs    []*indexJob                // newest last; running + recent finished
	projectLocks = map[string]*sync.Mutex{} // serializes directory jobs per project
	indexJobSeq  atomic.Int64
)

type indexJob struct {
	ID          string
	Path        string
	ProjectPath string
	StartedAt   time.Time
	filesDone   atomic.Int64
	symbols     atomic.Int64
	done        chan struct{}
	mu          sync.Mutex
	state       string
	finishedAt  time.Time
	err         string
}

func (j *indexJob) setState(state, errMsg string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.state = state
	j.err = errMsg
	if state == indexJobCompleted || state == indexJobFailed {
		j.finishedAt = time.Now()
	}
}

func (j *indexJob) active() bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.state == indexJobQueued || j.state == indexJobRunning
}

// snapshot is the JSON shape reported by index_files and index_status.
func (j *indexJob) snapshot() map[string]interface{} {
	j.mu.Lock()
	state, errMsg, finished := j.state, j.err, j.finishedAt
	j.mu.Unlock()
	end := time.Now()
	if !finished.IsZero() {
		end = finished
	}
	out := map[string]interface{}{
		"job_id":          j.ID,
		"status":          state,
		"path":            j.Path,
		"project_path":    j.ProjectPath,
		"files_done":      j.filesDone.Load(),
		"symbols_indexed": j.symbols.Load(),
		"started_at":      j.StartedAt.Format(time.RFC3339),
		"elapsed_ms":      end.Sub(j.StartedAt).Milliseconds(),
	}
	if !finished.IsZero() {
		out["finished_at"] = finished.Format(time.RFC3339)
	}
	if errMsg != "" {
		out["error"] = errMsg
	}
	return out
}

// pathWithin reports whether path is dir or lies under it.
func pathWithin(path, dir string) bool {
	if path == dir {
		return true
	}
	rel, err := filepath.Rel(dir, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// startIndexJob returns the active job already covering dirPath for this project
// (deduped=true) or starts a new one. Jobs for the same project run one at a time.
func startIndexJob(dirPath, projectPath string) (job *indexJob, deduped bool) {
	if abs, err := filepath.Abs(dirPath); err == nil {
		dirPath = abs
	}
	dirPath = filepath.Clean(dirPath)
	indexJobsMu.Lock()
	defer indexJobsMu.Unlock()
	for _, j := range indexJobs {
		if j.ProjectPath == projectPath && j.active() && pathWithin(dirPath, j.Path) {
			return j, true
		}
	}
	job = &indexJob{
		ID:          fmt.Sprintf("idx_%d_%d", time.Now().Unix(), indexJobSeq.Add(1)),
		Path:        dirPath,
		ProjectPath: projectPath,
		StartedAt:   time.Now(),
		done:        make(chan struct{}),
		state:       indexJobQueued,
	}
	indexJobs = append(indexJobs, job)
	pruneIndexJobsLocked()
	lock, ok := projectLocks[projectPath]
	if !ok {
		lock = &sync.Mutex{}
		projectLocks[projectPath] = lock
	}
	go runIndexJob(job, lock)
	return job, false
}

func runIndexJob(job *indexJob, lock *sync.Mutex) {
	defer close(job.done)
	defer func() {
		if r := recover(); r != nil {
			logger.Error("Index job panicked", "job_id", job.ID, "panic", r)
			job.setState(indexJobFailed, fmt.Sprintf("panic: %v", r))
		}
	}()
	lock.Lock()
	defer lock.Unlock()
	job.setState(indexJobRunning, "")
	n, err := runIndexDirectory(job.Path, job.ProjectPath, func(symbols int) {
		job.filesDone.Add(1)
		job.symbols.Add(int64(symbols))
	})
	job.symbols.Store(int64(n))
	if err != nil {
		logger.Warn("Index job failed", "job_id", job.ID, "path", job.Path, "files_done", job.filesDone.Load(), "error", err)
		job.setState(indexJobFailed, err.Error())
		return
	}
	afterIndexDirectory(job.ProjectPath)
	job.setState(indexJobCompleted, "")
	logger.Info("Index job completed", "job_id", job.ID, "path", job.Path, "files", job.filesDone.Load(), "symbols", n, "duration", time.Since(job.StartedAt).Round(time.Millisecond))
}

func defaultAfterIndexDirectory(projectPath string) {
	projectmeta.ClearDeleted(projectPath)
	embedqueue.UnmarkProjectCancelled(projectPath)
	watcher.EnsureWatcher(projectPath)
	if emb != nil {
		go embedqueue.EnqueueAllSymbolsFiles(projectPath)
	}
}

// pruneIndexJobsLocked drops the oldest finished jobs beyond maxFinishedIndexJobs.
func pruneIndexJobsLocked() {
	finished := 0
	for _, j := range indexJobs {
		if !j.active() {
			finished++
		}
	}
	if finished <= maxFinishedIndexJobs {
		return
	}
	drop := finished - maxFinishedIndexJobs
	kept := indexJobs[:0]
	for _, j := range indexJobs {
		if drop > 0 && !j.active() {
			drop--
			continue
		}
		kept = append(kept, j)
	}
	indexJobs = kept
}

// indexJobsForProject returns snapshots of this project's jobs, newest first.
func indexJobsForProject(projectPath string) []map[string]interface{} {
	indexJobsMu.Lock()
	var jobs []*indexJob
	for _, j := range indexJobs {
		if projectPath == "" || j.ProjectPath == projectPath {
			jobs = append(jobs, j)
		}
	}
	indexJobsMu.Unlock()
	sort.SliceStable(jobs, func(a, b int) bool { return jobs[a].StartedAt.After(jobs[b].StartedAt) })
	if len(jobs) > maxStatusIndexJobs {
		jobs = jobs[:maxStatusIndexJobs]
	}
	out := make([]map[string]interface{}, 0, len(jobs))
	for _, j := range jobs {
		out = append(out, j.snapshot())
	}
	return out
}

// handleIndexDirectory starts (or joins) a directory job and waits up to
// indexJobSyncWait. A finished job answers like the old synchronous call ("indexed");
// an unfinished one returns its status plus a poll hint.
func handleIndexDirectory(dirPath, projectPath string) map[string]interface{} {
	job, deduped := startIndexJob(dirPath, projectPath)
	select {
	case <-job.done:
	case <-time.After(indexJobSyncWait):
	}
	out := job.snapshot()
	if deduped {
		out["already_running"] = true
	}
	switch out["status"] {
	case indexJobCompleted:
		out["indexed"] = job.symbols.Load()
	case indexJobFailed:
		// "error" is already set, so the call reports isError.
	default:
		out["poll"] = fmt.Sprintf("indexing continues in the background; call index_status(project_path=%q) and check index_jobs for job %s", projectPath, job.ID)
	}
	return out
}

func withLinkedProjects(out map[string]interface{}, projectPath string) map[string]interface{} {
	if linked, _ := projectlinks.Links(projectPath); len(linked) > 0 {
		out["linked_projects"] = linked
	}
	return out
}

// indexStatusResult is index_status: symbol/file counts with resource and watcher
// health, plus index_files jobs and, when the data volume is low on space, a
// disk_pressure block explaining throttled embedding.
func indexStatusResult(projectPath string) map[string]interface{} {
	stats, err := indexer.GetIndexStats(projectPath)
	out := withResourceHealth(stats, err, projectPath)
	if jobs := indexJobsForProject(projectPath); len(jobs) > 0 {
		out["index_jobs"] = jobs
		for _, j := range jobs {
			if s := j["status"]; s == indexJobQueued || s == indexJobRunning {
				out["indexing"] = true
				break
			}
		}
	}
	if d := db.SampleDiskSpace(); d.Level != db.DiskOK {
		out["disk_pressure"] = map[string]interface{}{
			"level":      d.Level,
			"free_bytes": d.FreeBytes,
			"free":       db.FormatFileSize(int64(d.FreeBytes)),
			"data_dir":   d.Path,
			"effect":     "embedding is throttled (low) or paused (critical) until space is freed on the data directory's volume",
		}
	}
	return out
}
