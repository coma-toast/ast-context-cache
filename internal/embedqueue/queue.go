package embedqueue

import (
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// ring buffer for throughput (last 5 seconds, 10 slots @ 500ms)
const (
	pendingCap       = 128
	highCap          = 128
	lowCap           = 2048
	throughputSlots  = 10
	throughputWindow = 500 * time.Millisecond
)

type job struct {
	file, projectPath string
}

// ActivityEntry is one embed queue activity row (file + owning project).
type ActivityEntry struct {
	File        string
	ProjectPath string
}

var (
	emb           embedder.Interface
	embMu         sync.RWMutex
	pendingCh     chan job
	highCh        chan job
	lowCh         chan job
	started       sync.Once
	inFlight     int64
	completed    int64
	failed       int64
	throughput   [throughputSlots]int64
	lastSlot     int64
	startedAt    time.Time

	// Recent activity log (last 20 embeddings)
	recentActivity      [20]ActivityEntry
	recentActivityIdx   int
	recentActivityCount int
	recentActivityMu    sync.Mutex

	activeMu        sync.Mutex
	activeJobs      map[string]struct{}
	activeProjects  map[string]string

	pendingMu        sync.Mutex
	pending          map[string]job
	pendingChQueued  map[string]struct{}
	drainMu          sync.Mutex
	pendingPeak      atomic.Int64
)

// Ready reports whether the embed queue workers are running.
func Ready() bool {
	return highCh != nil
}

// SetEmbedder swaps the embedder used by queue workers (hot reload).
func SetEmbedder(e embedder.Interface) {
	embMu.Lock()
	emb = e
	embMu.Unlock()
}

func queueEmbedder() embedder.Interface {
	embMu.RLock()
	defer embMu.RUnlock()
	return emb
}

// Start launches worker goroutines; safe to call once.
func Start(e embedder.Interface) {
	started.Do(func() {
		SetEmbedder(e)
		pendingCh = make(chan job, pendingCap)
		highCh = make(chan job, highCap)
		lowCh = make(chan job, lowCap)
		workerStop = make(chan struct{}, AbsoluteMaxWorkers)
		beginProcessingWindow()
	workerMu.Lock()
	workerCount = loadWorkerCount()
	workerTarget = workerCount
	n := workerCount
	workerMu.Unlock()
		for i := 0; i < n; i++ {
			go worker()
		}
		startedAt = time.Now()
		embedder.SetProbeDeferCheck(func() bool {
			return Snapshot().InFlight > 0
		})
		embedder.SetOnRecovery(func() { RecoverAfterEmbedder() })
		log.Printf("embed queue: %d workers (pending=%d high=%d low=%d)", n, pendingCap, highCap, lowCap)
		startPendingDBFlusher()
		LoadPendingFromDB()
		go flushPendingIfReady()
		startPressureBackoff()
	})
}

func worker() {
	workerLive.Add(1)
	defer func() {
		workerLive.Add(-1)
		notifyWorkerPoolLiveChange()
	}()
	waitForProcessingReady()
	for {
		select {
		case <-workerStop:
			return
		default:
		}
		select {
		case <-workerStop:
			return
		case j := <-pendingCh:
			run(j)
		default:
			select {
			case <-workerStop:
				return
			case j := <-pendingCh:
				run(j)
			case j := <-highCh:
				run(j)
			case j := <-lowCh:
				run(j)
			}
		}
	}
}

func run(j job) {
	runWithEmbedder(j, queueEmbedder())
}

func runWithEmbedder(j job, e embedder.Interface) {
	atomic.AddInt64(&inFlight, 1)
	trackJobStart(j.file, j.projectPath)
	realtime.Notify(realtime.EmbedFinished)
	defer func() {
		unmarkPendingChQueued(j)
		trackJobEnd(j.file)
		atomic.AddInt64(&inFlight, -1)
		realtime.Notify(realtime.EmbedFinished)
	}()
	if db.IndexReadQuiesced() {
		markPending(j)
		return
	}
	if isProjectCancelled(j.projectPath) {
		return
	}
	if e == nil {
		return
	}
	if e == queueEmbedder() {
		if state, _ := embedder.HealthState(); state == "error" {
			markPending(j)
			return
		}
	}
	if err := indexer.EmbedFileSymbols(e, j.file, j.projectPath); err != nil {
		atomic.AddInt64(&failed, 1)
		markPending(j)
		return
	}
	clearPending(j)
	atomic.AddInt64(&completed, 1)
	recordActivity(j.file, j.projectPath)

	slot := int(time.Since(startedAt)/throughputWindow) % throughputSlots
	atomic.AddInt64(&throughput[slot], 1)
}

// Submit enqueues a low-priority embed for one file.
func Submit(file, projectPath string) {
	SubmitPriority(file, projectPath, false)
}

// SubmitPriority enqueues an embed; high priority is used for pinned projects.
// Start must have been called from main with a non-nil embedder.
func SubmitPriority(file, projectPath string, high bool) {
	if isProjectCancelled(projectPath) {
		return
	}
	j := job{file: file, projectPath: projectPath}
	if MaintenancePaused() {
		if markPendingIfNew(j, pendingReasonDrained) {
			realtime.Notify(realtime.EmbedFinished)
		}
		return
	}
	if highCh == nil {
		e := queueEmbedder()
		if e != nil {
			go func() {
				trackJobStart(file, projectPath)
				realtime.Notify(realtime.EmbedFinished)
				defer func() {
					trackJobEnd(file)
					realtime.Notify(realtime.EmbedFinished)
				}()
				if err := indexer.EmbedFileSymbols(e, file, projectPath); err != nil {
					atomic.AddInt64(&failed, 1)
					markPending(job{file: file, projectPath: projectPath})
					return
				}
				clearPending(job{file: file, projectPath: projectPath})
				atomic.AddInt64(&completed, 1)
				recordActivity(file, projectPath)
			}()
		}
		return
	}
	if state, _ := embedder.HealthState(); state == "error" {
		if markPendingIfNew(j, pendingReasonDrained) {
			realtime.Notify(realtime.EmbedFinished)
		}
		return
	}
	if high {
		select {
		case highCh <- j:
			realtime.Notify(realtime.EmbedFinished)
		default:
			select {
			case highCh <- j:
				realtime.Notify(realtime.EmbedFinished)
			case lowCh <- j:
				realtime.Notify(realtime.EmbedFinished)
			}
		}
		return
	}
	lowCh <- j
	realtime.Notify(realtime.EmbedFinished)
}

// EnqueueAllSymbolsFiles enqueues an embed job for every indexed file in the project (e.g. after full directory index).
func EnqueueAllSymbolsFiles(projectPath string) {
	rows, err := db.IndexDB.Query(
		"SELECT DISTINCT file FROM symbols WHERE project_path = ?", projectPath)
	if err != nil {
		log.Printf("embedqueue: list files: %v", err)
		return
	}
	defer rows.Close()
	high := db.IsPinnedProject(projectPath)
	for rows.Next() {
		var f string
		rows.Scan(&f)
		SubmitPriority(f, projectPath, high)
	}
}

// ThroughputLast5s returns average embeddings per second over the last 5 seconds.
func ThroughputLast5s() int64 {
	if startedAt.IsZero() {
		return 0
	}
	var total int64
	currentSlot := int(time.Since(startedAt) / throughputWindow)
	for i := 0; i < throughputSlots; i++ {
		idx := (currentSlot - i + throughputSlots*1024) % throughputSlots
		total += atomic.LoadInt64(&throughput[idx])
	}
	return total / int64(throughputSlots/2)
}

// QueueSnapshot is queue depth and worker state for dashboards.
type QueueSnapshot struct {
	Queued      int
	Pending     int
	PendingPeak int
	HighUsed    int
	LowUsed     int
	HighCap     int
	LowCap      int
	Workers         int
	WorkersEffective int
	WorkersLive     int
	AuxWorkers          int
	AuxWorkersEffective int
	AuxWorkersLive      int
	InFlight    int64
	Completed   int64
	Failed      int64
	Throughput  int64
}

// InFlight returns embed jobs currently running.
func InFlight() int64 {
	return atomic.LoadInt64(&inFlight)
}

// Snapshot returns current queue and worker metrics.
func Snapshot() QueueSnapshot {
	s := QueueSnapshot{HighCap: highCap, LowCap: lowCap, Workers: WorkerTarget(), WorkersEffective: WorkerCount(), WorkersLive: WorkerLive(), AuxWorkers: AuxWorkerTarget(), AuxWorkersEffective: AuxWorkerCount(), AuxWorkersLive: AuxWorkerLive()}
	s.InFlight = atomic.LoadInt64(&inFlight)
	s.Completed = atomic.LoadInt64(&completed)
	s.Failed = atomic.LoadInt64(&failed)
	s.Pending = PendingCount()
	s.PendingPeak = PendingPeak()
	s.Throughput = ThroughputLast5s()
	if highCh == nil {
		return s
	}
	s.HighUsed = len(highCh)
	s.LowUsed = len(lowCh)
	s.Queued = s.HighUsed + s.LowUsed
	return s
}

func jobKey(j job) string {
	return j.file + "\x00" + j.projectPath
}

func markPending(j job) {
	if markPendingIfNew(j, pendingReasonFailed) {
		realtime.Notify(realtime.EmbedFinished)
		flushPendingIfReady()
	}
}

func clearPending(j job) {
	pendingMu.Lock()
	if len(pending) > 0 {
		delete(pending, jobKey(j))
	}
	pendingMu.Unlock()
	clearPendingDB(j)
	realtime.Notify(realtime.EmbedFinished)
}

// PendingCount is files that failed embedding and await retry.
func PendingCount() int {
	pendingMu.Lock()
	n := len(pending)
	pendingMu.Unlock()
	trackPendingPeak(n)
	return n
}

// PendingPeak is the highest pending count since the last time pending was zero.
func PendingPeak() int {
	pendingMu.Lock()
	n := len(pending)
	pendingMu.Unlock()
	if n == 0 {
		return 0
	}
	peak := int(pendingPeak.Load())
	if peak < n {
		peak = n
	}
	return peak
}

func trackPendingPeak(n int) {
	if n == 0 {
		pendingPeak.Store(0)
		return
	}
	for {
		cur := pendingPeak.Load()
		if int64(n) <= cur {
			return
		}
		if pendingPeak.CompareAndSwap(cur, int64(n)) {
			return
		}
	}
}

func enqueuePendingRetry(j job) bool {
	if MaintenancePaused() {
		return false
	}
	k := jobKey(j)
	pendingMu.Lock()
	if pending == nil {
		pendingMu.Unlock()
		return false
	}
	if _, ok := pending[k]; !ok {
		pendingMu.Unlock()
		return false
	}
	if pendingChQueued == nil {
		pendingChQueued = map[string]struct{}{}
	}
	if _, ok := pendingChQueued[k]; ok {
		pendingMu.Unlock()
		return false
	}
	pendingChQueued[k] = struct{}{}
	pendingMu.Unlock()
	if pendingCh == nil {
		unmarkPendingChQueued(j)
		return false
	}
	select {
	case pendingCh <- j:
		realtime.Notify(realtime.EmbedFinished)
		return true
	default:
		unmarkPendingChQueued(j)
		return false
	}
}

func unmarkPendingChQueued(j job) {
	pendingMu.Lock()
	delete(pendingChQueued, jobKey(j))
	pendingMu.Unlock()
}

func drainCh(ch chan job, reason string, clearPendingQueued bool) int {
	if ch == nil {
		return 0
	}
	n := 0
	for {
		select {
		case j := <-ch:
			if clearPendingQueued {
				unmarkPendingChQueued(j)
			}
			markPendingIfNew(j, reason)
			n++
		default:
			return n
		}
	}
}

// DrainQueueToPending moves all queued embed jobs into pending when the embedder is down.
func DrainQueueToPending() int {
	drainMu.Lock()
	defer drainMu.Unlock()
	n := drainCh(pendingCh, pendingReasonDrained, true)
	n += drainCh(highCh, pendingReasonDrained, false)
	n += drainCh(lowCh, pendingReasonDrained, false)
	return n
}

// OnEmbedderError drains in-flight queue work into pending so the dashboard reflects backlog.
func OnEmbedderError() {
	if n := DrainQueueToPending(); n > 0 {
		log.Printf("embedqueue: drained %d queued jobs to pending", n)
		realtime.Notify(realtime.EmbedFinished)
	}
}

// FlushPending re-queues all pending embed jobs (e.g. after embedder recovery).
func FlushPending() {
	pendingMu.Lock()
	if len(pending) == 0 {
		pendingMu.Unlock()
		return
	}
	jobs := make([]job, 0, len(pending))
	for _, j := range pending {
		jobs = append(jobs, j)
	}
	pendingMu.Unlock()
	s := Snapshot()
	log.Printf("embedqueue: flushing %d pending queued=%d inFlight=%d", len(jobs), s.Queued, s.InFlight)
	for _, j := range jobs {
		enqueuePendingRetry(j)
	}
}

// Stats returns queued jobs, active workers, total workers, completed, and throughput.
func Stats() (queued int, active int64, totalWorkers int, completed int64, throughput int64) {
	s := Snapshot()
	return s.Queued, s.InFlight, s.Workers, s.Completed, s.Throughput
}

func recordActivity(file, projectPath string) {
	recentActivityMu.Lock()
	defer recentActivityMu.Unlock()
	recentActivity[recentActivityIdx] = ActivityEntry{File: file, ProjectPath: projectPath}
	recentActivityIdx = (recentActivityIdx + 1) % 20
	if recentActivityCount < 20 {
		recentActivityCount++
	}
}

func trackJobStart(file, projectPath string) {
	activeMu.Lock()
	if activeJobs == nil {
		activeJobs = map[string]struct{}{}
	}
	if activeProjects == nil {
		activeProjects = map[string]string{}
	}
	activeJobs[file] = struct{}{}
	if projectPath != "" {
		activeProjects[file] = projectPath
	}
	activeMu.Unlock()
}

func trackJobEnd(file string) {
	activeMu.Lock()
	delete(activeJobs, file)
	delete(activeProjects, file)
	activeMu.Unlock()
}

// CurrentJobs returns files currently being embedded (sorted, newest activity order undefined).
func CurrentJobs() []ActivityEntry {
	activeMu.Lock()
	defer activeMu.Unlock()
	if len(activeJobs) == 0 {
		return nil
	}
	out := make([]ActivityEntry, 0, len(activeJobs))
	for f := range activeJobs {
		out = append(out, ActivityEntry{File: f, ProjectPath: activeProjects[f]})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].File < out[j].File })
	return out
}

// RecentActivity returns the most recent embedding activity (newest first).
func RecentActivity() []ActivityEntry {
	recentActivityMu.Lock()
	defer recentActivityMu.Unlock()
	if recentActivityCount == 0 {
		return nil
	}
	result := make([]ActivityEntry, recentActivityCount)
	for i := 0; i < recentActivityCount; i++ {
		idx := (recentActivityIdx - 1 - i + 20) % 20
		result[i] = recentActivity[idx]
	}
	return result
}
