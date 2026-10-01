package db

import (
	"database/sql"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
)

// errIndexWriterStopped is returned to an IndexWrite whose job was handed to a
// writer that stopped (quiesce or Close) before running it.
var errIndexWriterStopped = errors.New("index writer stopped before running write")

type indexWriteJob struct {
	fn   func(*sql.Tx) error
	done chan error
	// claimed is won exactly once: by the writer just before it runs fn (it then
	// always replies on done), or by the caller abandoning the job after the
	// writer's stop channel closed. Without it a job buffered in a stopped
	// writer's channel is never read, and its caller blocks on done forever.
	claimed atomic.Bool
}

func (j *indexWriteJob) claim() bool { return j.claimed.CompareAndSwap(false, true) }

var (
	indexWriteCh    chan *indexWriteJob
	indexWriterMu   sync.Mutex
	indexWriterStop chan struct{}
)

func startIndexWriter() {
	indexWriterMu.Lock()
	defer indexWriterMu.Unlock()
	// Don't resurrect the writer mid-quiesce: a caller that passed IndexWrite's
	// gate check just before quiesceIndexPool stopped the writer would otherwise
	// start a fresh one against the pool being closed. restoreIndexPool starts
	// the next writer.
	if indexWriterStop != nil || IndexReadQuiesced() {
		return
	}
	indexWriteCh = make(chan *indexWriteJob, 512)
	indexWriterStop = make(chan struct{})
	go runIndexWriter(indexWriteCh, indexWriterStop)
}

func resetIndexWriter() {
	indexWriterMu.Lock()
	defer indexWriterMu.Unlock()
	if indexWriterStop != nil {
		select {
		case <-indexWriterStop:
		default:
			close(indexWriterStop)
		}
	}
	indexWriteCh = make(chan *indexWriteJob, 512)
	indexWriterStop = make(chan struct{})
	go runIndexWriter(indexWriteCh, indexWriterStop)
}

func stopIndexWriter() {
	indexWriterMu.Lock()
	defer indexWriterMu.Unlock()
	if indexWriterStop == nil {
		return
	}
	select {
	case <-indexWriterStop:
		return
	default:
		close(indexWriterStop)
	}
	indexWriterStop = nil
	indexWriteCh = nil
}

// runIndexWriter takes its channels as parameters, captured once at start
// time, rather than reading the mutable indexWriteCh/indexWriterStop package
// vars directly: stopIndexWriter/resetIndexWriter reassign those vars under
// indexWriterMu, but this goroutine never held that lock, racing on every read.
func runIndexWriter(ch chan *indexWriteJob, stop chan struct{}) {
	for {
		select {
		case <-stop:
			return
		case job, ok := <-ch:
			if !ok {
				return
			}
			// select picks at random when stop and a buffered job are both ready.
			// Once stop is closed, leave the job unclaimed for its caller to
			// abandon rather than writing to a pool that is being closed.
			select {
			case <-stop:
				return
			default:
			}
			if job.claim() {
				job.done <- indexWriteTx(job.fn)
			}
		}
	}
}

func indexWriteTx(fn func(*sql.Tx) error) error {
	if IndexDB == nil {
		return fmt.Errorf("index db unavailable")
	}
	tx, err := IndexDB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit()
}

// IndexWrite serializes index mutations on a single writer goroutine.
func IndexWrite(fn func(*sql.Tx) error) error {
	if IndexReadQuiesced() {
		return fmt.Errorf("index db quiesced for maintenance")
	}
	startIndexWriter()
	indexWriterMu.Lock()
	ch := indexWriteCh
	stop := indexWriterStop
	indexWriterMu.Unlock()
	if ch == nil {
		// Stopped (quiesce or Close) since startIndexWriter; the pool is going away.
		return errIndexWriterStopped
	}
	return submitIndexWrite(ch, stop, fn)
}

// submitIndexWrite hands fn to the writer that owns ch and waits for its result.
// ch and stop may belong to a writer that has stopped since the caller read
// them, so every wait also watches stop: once it closes, the job is either
// abandoned (unclaimed) or already running and about to reply, never stranded.
func submitIndexWrite(ch chan *indexWriteJob, stop chan struct{}, fn func(*sql.Tx) error) error {
	select {
	case <-stop:
		return errIndexWriterStopped
	default:
	}
	job := &indexWriteJob{fn: fn, done: make(chan error, 1)}
	select {
	case ch <- job:
	case <-stop:
		return errIndexWriterStopped
	}
	select {
	case err := <-job.done:
		return err
	case <-stop:
		if job.claim() {
			return errIndexWriterStopped
		}
		// The writer claimed it first, so it is running fn and will reply.
		return <-job.done
	}
}

// FlushIndexWriter drains pending index writes (call before WAL checkpoint).
func FlushIndexWriter() {
	indexWriterMu.Lock()
	ch := indexWriteCh
	stop := indexWriterStop
	indexWriterMu.Unlock()
	if ch == nil || stop == nil {
		return
	}
	_ = submitIndexWrite(ch, stop, func(*sql.Tx) error { return nil })
}
