package embedqueue

import (
	"log"
	"sync"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

const (
	workers = 3
	highCap = 128
	lowCap  = 2048
)

type job struct {
	file, projectPath string
}

var (
	emb      embedder.Interface
	highCh   chan job
	lowCh    chan job
	started  sync.Once
	inFlight int64
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
	defer atomic.AddInt64(&inFlight, -1)
	if emb == nil {
		return
	}
	indexer.EmbedFileSymbols(emb, j.file, j.projectPath)
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
			indexer.EmbedFileSymbols(emb, file, projectPath)
		}
		return
	}
	j := job{file: file, projectPath: projectPath}
	if high {
		select {
		case highCh <- j:
		default:
			select {
			case highCh <- j:
			case lowCh <- j:
			}
		}
		return
	}
	lowCh <- j
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

// Stats returns queued jobs (waiting in channels) and active embedding workers.
func Stats() (queued int, active int64) {
	if highCh == nil {
		return 0, atomic.LoadInt64(&inFlight)
	}
	return len(highCh) + len(lowCh), atomic.LoadInt64(&inFlight)
}
