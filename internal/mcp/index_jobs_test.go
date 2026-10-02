package mcp

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// isolateIndexJobs gives a test fresh job state and restores the package hooks once
// every job it started has finished. HOME and the DB are isolated by TestMain.
func isolateIndexJobs(t *testing.T) {
	t.Helper()
	origCfg, origRun, origAfter, origWait := srvCfg, runIndexDirectory, afterIndexDirectory, indexJobSyncWait
	srvCfg = DefaultConfig()
	indexJobsMu.Lock()
	indexJobs = nil
	projectLocks = map[string]*sync.Mutex{}
	indexJobsMu.Unlock()
	afterIndexDirectory = func(string) {}
	t.Cleanup(func() {
		indexJobsMu.Lock()
		jobs := append([]*indexJob(nil), indexJobs...)
		indexJobsMu.Unlock()
		for _, j := range jobs {
			select {
			case <-j.done:
			case <-time.After(10 * time.Second):
				t.Errorf("job %s still running at cleanup", j.ID)
			}
		}
		srvCfg, runIndexDirectory, afterIndexDirectory, indexJobSyncWait = origCfg, origRun, origAfter, origWait
		indexJobsMu.Lock()
		indexJobs = nil
		indexJobsMu.Unlock()
	})
}

// blockingRunner simulates a slow directory index: it reports `files` files, then
// blocks until release is closed. calls/maxConcurrent let tests detect double-indexing.
type blockingRunner struct {
	release       chan struct{}
	files         int
	calls         atomic.Int32
	running       atomic.Int32
	maxConcurrent atomic.Int32
}

func newBlockingRunner(files int) *blockingRunner {
	return &blockingRunner{release: make(chan struct{}), files: files}
}

func (b *blockingRunner) run(dir, project string, onFile func(int)) (int, error) {
	b.calls.Add(1)
	n := b.running.Add(1)
	defer b.running.Add(-1)
	for {
		m := b.maxConcurrent.Load()
		if n <= m || b.maxConcurrent.CompareAndSwap(m, n) {
			break
		}
	}
	for i := 0; i < b.files; i++ {
		onFile(10)
	}
	<-b.release
	return b.files * 10, nil
}

func waitIndexJobs(t *testing.T) {
	t.Helper()
	indexJobsMu.Lock()
	jobs := append([]*indexJob(nil), indexJobs...)
	indexJobsMu.Unlock()
	for _, j := range jobs {
		select {
		case <-j.done:
		case <-time.After(10 * time.Second):
			t.Fatalf("job %s did not finish", j.ID)
		}
	}
}

// Field report #10: index_files on a directory blocked until the MCP client timed out.
// It must now return promptly with a job the agent can poll via index_status.
func TestIndexFilesDirectoryReturnsPromptlyWithPollableJob(t *testing.T) {
	isolateIndexJobs(t)
	runner := newBlockingRunner(3)
	runIndexDirectory = runner.run
	indexJobSyncWait = 50 * time.Millisecond
	project := t.TempDir()

	start := time.Now()
	out, isErr := callTool(t, "index_files", map[string]interface{}{"path": project, "project_path": project})
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("index_files blocked for %s on a slow directory", elapsed)
	}
	if isErr {
		t.Fatalf("isError=true for a running job: %v", out)
	}
	if st := out["status"]; (st != indexJobRunning && st != indexJobQueued) || out["job_id"] == "" || out["poll"] == nil {
		t.Fatalf("want queued/running job with job_id and poll hint, got %v", out)
	}
	if _, ok := out["indexed"]; ok {
		t.Fatalf("unfinished job must not report indexed: %v", out)
	}

	// Progress is visible through index_status while the job is still running.
	deadline := time.Now().Add(5 * time.Second)
	for {
		status, _ := callTool(t, "index_status", map[string]interface{}{"project_path": project})
		jobs, _ := status["index_jobs"].([]interface{})
		if len(jobs) != 1 || status["indexing"] != true {
			t.Fatalf("index_status should list the running job: %v", status)
		}
		j := jobs[0].(map[string]interface{})
		if j["job_id"] != out["job_id"] {
			t.Fatalf("index_status job=%v want %v", j, out["job_id"])
		}
		if j["status"] == indexJobRunning && j["files_done"] == float64(3) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("progress never reached 3 files: %v", j)
		}
		time.Sleep(10 * time.Millisecond)
	}

	close(runner.release)
	waitIndexJobs(t)
	status, _ := callTool(t, "index_status", map[string]interface{}{"project_path": project})
	j := status["index_jobs"].([]interface{})[0].(map[string]interface{})
	if j["status"] != indexJobCompleted || j["symbols_indexed"] != float64(30) || j["finished_at"] == nil {
		t.Fatalf("after completion index_status job=%v", j)
	}
	if _, ok := status["indexing"]; ok {
		t.Fatalf("indexing flag should clear once jobs finish: %v", status)
	}
}

// Concurrent index_files calls on the same directory (or a subdirectory of one already
// being indexed) must join the running job instead of indexing twice.
func TestIndexFilesConcurrentCallsShareOneJob(t *testing.T) {
	isolateIndexJobs(t)
	runner := newBlockingRunner(1)
	runIndexDirectory = runner.run
	indexJobSyncWait = 20 * time.Millisecond
	project := t.TempDir()
	sub := filepath.Join(project, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	ids := make([]string, 8)
	for i := range ids {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			dir := project
			if i%2 == 1 {
				dir = sub
			}
			ids[i], _ = handleIndexDirectory(dir, project)["job_id"].(string)
		}(i)
	}
	wg.Wait()
	close(runner.release)
	waitIndexJobs(t)
	// The first caller might have been the subdirectory; then parent calls queue one
	// more job. Either way the same directory is never indexed twice concurrently.
	if c := runner.calls.Load(); c > 2 {
		t.Fatalf("runner called %d times for 8 overlapping requests", c)
	}
	if m := runner.maxConcurrent.Load(); m != 1 {
		t.Fatalf("max concurrent directory index runs = %d, want 1", m)
	}
	unique := map[string]bool{}
	for _, id := range ids {
		unique[id] = true
	}
	if len(unique) != int(runner.calls.Load()) {
		t.Fatalf("job ids %v do not match %d runs", ids, runner.calls.Load())
	}

	// Same directory, sequential calls while running: exactly one run.
	runner2 := newBlockingRunner(0)
	runIndexDirectory = runner2.run
	a := handleIndexDirectory(project, project)
	b := handleIndexDirectory(project, project)
	close(runner2.release)
	waitIndexJobs(t)
	if a["job_id"] != b["job_id"] || b["already_running"] != true || runner2.calls.Load() != 1 {
		t.Fatalf("second call should join the first: a=%v b=%v calls=%d", a, b, runner2.calls.Load())
	}
}

// A directory job that finishes within the sync wait keeps the old synchronous answer.
func TestIndexFilesSmallDirectoryCompletesSynchronously(t *testing.T) {
	isolateIndexJobs(t)
	project := t.TempDir()
	src := "package demo\n\nfunc Hello() string { return \"hi\" }\n\nfunc World() {}\n"
	if err := os.WriteFile(filepath.Join(project, "demo.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	out, isErr := callTool(t, "index_files", map[string]interface{}{"path": project, "project_path": project})
	if isErr || out["status"] != indexJobCompleted {
		t.Fatalf("want completed, got isErr=%v %v", isErr, out)
	}
	if n, _ := out["indexed"].(float64); n < 2 {
		t.Fatalf("indexed=%v want >= 2 symbols", out["indexed"])
	}
	if out["files_done"] != float64(1) {
		t.Fatalf("files_done=%v want 1", out["files_done"])
	}
}

func TestIndexFilesDirectoryFailureReportsError(t *testing.T) {
	isolateIndexJobs(t)
	runIndexDirectory = func(string, string, func(int)) (int, error) { return 0, errors.New("walk: permission denied") }
	project := t.TempDir()
	out, isErr := callTool(t, "index_files", map[string]interface{}{"path": project, "project_path": project})
	if !isErr || out["status"] != indexJobFailed || out["error"] != "walk: permission denied" {
		t.Fatalf("want failed job with error, got isErr=%v %v", isErr, out)
	}
}

func TestPruneIndexJobsKeepsActiveAndRecent(t *testing.T) {
	isolateIndexJobs(t)
	indexJobsMu.Lock()
	for i := 0; i < maxFinishedIndexJobs+5; i++ {
		j := &indexJob{ID: "done", done: make(chan struct{}), state: indexJobCompleted}
		close(j.done)
		indexJobs = append(indexJobs, j)
	}
	active := &indexJob{ID: "active", done: make(chan struct{}), state: indexJobRunning}
	indexJobs = append([]*indexJob{active}, indexJobs...)
	pruneIndexJobsLocked()
	n := len(indexJobs)
	first := indexJobs[0]
	indexJobs = indexJobs[1:]
	indexJobsMu.Unlock()
	close(active.done)
	if n != maxFinishedIndexJobs+1 || first != active {
		t.Fatalf("kept %d jobs (first=%s), want %d with the active job retained", n, first.ID, maxFinishedIndexJobs+1)
	}
}

// Field report #12: low free space on the data volume (WAL growth during catch-up) must be
// visible to the agent in index_status, not only in the server log — and clear once space
// is back. The free-space probe is faked; nothing fills a disk.
func TestIndexStatusReportsDiskPressure(t *testing.T) {
	isolateIndexJobs(t)
	free := uint64(3 << 30)
	restore := db.SetFreeBytesFuncForTest(func(string) (uint64, error) { return free, nil })
	t.Cleanup(restore)
	project := t.TempDir()

	for _, tc := range []struct {
		free  uint64
		level string
	}{{3 << 30, db.DiskLow}, {500 << 20, db.DiskCritical}} {
		free = tc.free
		status, isErr := callTool(t, "index_status", map[string]interface{}{"project_path": project})
		dp, ok := status["disk_pressure"].(map[string]interface{})
		if isErr || !ok {
			t.Fatalf("free=%d MiB: want disk_pressure in index_status, got isErr=%v %v", tc.free>>20, isErr, status)
		}
		if dp["level"] != tc.level || dp["free_bytes"] != float64(tc.free) || dp["effect"] == nil {
			t.Fatalf("free=%d MiB: disk_pressure=%v want level %s", tc.free>>20, dp, tc.level)
		}
	}

	free = 100 << 30
	status, _ := callTool(t, "index_status", map[string]interface{}{"project_path": project})
	if dp, ok := status["disk_pressure"]; ok {
		t.Fatalf("disk_pressure should clear once space is back: %v", dp)
	}
}
