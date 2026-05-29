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
	workers       = 3
	highCap       = 128
	lowCap        = 2048
	throughputSlots = 10
	throughputWindow = 500 * time.Millisecond
)

type job struct {
	file, projectPath string
}

var (
	emb           embedder.Interface
	highCh        chan job
	lowCh         chan job
	started       sync.Once
	inFlight     int64
	completed    int64
	throughput   [throughputSlots]int64
	lastSlot     int64
	startedAt    time.Time

	// Recent activity log (last 20 embeddings)
	recentActivity      [20]string
	recentActivityIdx   int
	recentActivityCount int
	recentActivityMu    sync.Mutex

	activeMu   sync.Mutex
	activeJobs map[string]struct{}
)

// Start launches worker goroutines; safe to call once.
func Start(e embedder.Interface) {
	started.Do(func() {
		emb = e
		highCh = make(chan job, highCap)
		lowCh = make(chan job, lowCap)
		for i := 0; i < workers; i++ {
			go worker()
		}
		startedAt = time.Now()
		log.Printf("embed queue: %d workers (high=%d low=%d)", workers, highCap, lowCap)
	})
}

func worker() {
	for {
		select {
		case j := <-highCh:
			run(j)
		default:
			select {
			case j := <-highCh:
				run(j)
			case j := <-lowCh:
				run(j)
			}
		}
	}
}

func run(j job) {
	atomic.AddInt64(&inFlight, 1)
	trackJobStart(j.file)
	realtime.Notify(realtime.EmbedFinished)
	defer func() {
		trackJobEnd(j.file)
		atomic.AddInt64(&inFlight, -1)
		realtime.Notify(realtime.EmbedFinished)
	}()
	if emb == nil {
		return
	}
	indexer.EmbedFileSymbols(emb, j.file, j.projectPath)
	atomic.AddInt64(&completed, 1)
	recordActivity(j.file)

	slot := int(time.Since(startedAt) / throughputWindow)
	if slot >= 0 && slot < throughputSlots {
		atomic.AddInt64(&throughput[slot%throughputSlots], 1)
	}
}

// Submit enqueues a low-priority embed for one file.
func Submit(file, projectPath string) {
	SubmitPriority(file, projectPath, false)
}

// SubmitPriority enqueues an embed; high priority is used for pinned projects.
// Start must have been called from main with a non-nil embedder.
func SubmitPriority(file, projectPath string, high bool) {
	if highCh == nil || emb == nil {
		if emb != nil {
			go func() {
				trackJobStart(file)
				realtime.Notify(realtime.EmbedFinished)
				defer func() {
					trackJobEnd(file)
					realtime.Notify(realtime.EmbedFinished)
				}()
				indexer.EmbedFileSymbols(emb, file, projectPath)
				atomic.AddInt64(&completed, 1)
				recordActivity(file)
			}()
		}
		return
	}
	j := job{file: file, projectPath: projectPath}
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
	rows, err := db.DB.Query(
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

// ThroughputLast5s returns embeddings processed in last 5 seconds.
func ThroughputLast5s() int64 {
	if startedAt.IsZero() {
		return 0
	}
	var total int64
	currentSlot := int(time.Since(startedAt) / throughputWindow)
	for i := 0; i < throughputSlots; i++ {
		slot := currentSlot - i
		if slot < 0 {
			slot += throughputSlots
		}
		total += atomic.LoadInt64(&throughput[slot%throughputSlots])
	}
	return total
}

// QueueSnapshot is queue depth and worker state for dashboards.
type QueueSnapshot struct {
	Queued      int
	HighUsed    int
	LowUsed     int
	HighCap     int
	LowCap      int
	Workers     int
	InFlight    int64
	Completed   int64
	Throughput  int64
}

// Snapshot returns current queue and worker metrics.
func Snapshot() QueueSnapshot {
	s := QueueSnapshot{HighCap: highCap, LowCap: lowCap, Workers: workers}
	s.InFlight = atomic.LoadInt64(&inFlight)
	s.Completed = atomic.LoadInt64(&completed)
	s.Throughput = ThroughputLast5s()
	if highCh == nil {
		return s
	}
	s.HighUsed = len(highCh)
	s.LowUsed = len(lowCh)
	s.Queued = s.HighUsed + s.LowUsed
	return s
}

// Stats returns queued jobs, active workers, total workers, completed, and throughput.
func Stats() (queued int, active int64, totalWorkers int, completed int64, throughput int64) {
	s := Snapshot()
	return s.Queued, s.InFlight, s.Workers, s.Completed, s.Throughput
}

func recordActivity(file string) {
	recentActivityMu.Lock()
	defer recentActivityMu.Unlock()
	recentActivity[recentActivityIdx] = file
	recentActivityIdx = (recentActivityIdx + 1) % 20
	if recentActivityCount < 20 {
		recentActivityCount++
	}
}

func trackJobStart(file string) {
	activeMu.Lock()
	if activeJobs == nil {
		activeJobs = map[string]struct{}{}
	}
	activeJobs[file] = struct{}{}
	activeMu.Unlock()
}

func trackJobEnd(file string) {
	activeMu.Lock()
	delete(activeJobs, file)
	activeMu.Unlock()
}

// CurrentJobs returns files currently being embedded (sorted).
func CurrentJobs() []string {
	activeMu.Lock()
	defer activeMu.Unlock()
	if len(activeJobs) == 0 {
		return nil
	}
	out := make([]string, 0, len(activeJobs))
	for f := range activeJobs {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// RecentActivity returns the most recent embedding activity (newest first).
func RecentActivity() []string {
	recentActivityMu.Lock()
	defer recentActivityMu.Unlock()
	if recentActivityCount == 0 {
		return nil
	}
	result := make([]string, recentActivityCount)
	for i := 0; i < recentActivityCount; i++ {
		idx := (recentActivityIdx - 1 - i + 20) % 20
		result[i] = recentActivity[idx]
	}
	return result
}
