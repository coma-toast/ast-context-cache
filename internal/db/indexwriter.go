package db

import (
	"database/sql"
	"fmt"
	"sync"
)

type indexWriteJob struct {
	fn   func(*sql.Tx) error
	done chan error
}

var (
	indexWriteCh    chan indexWriteJob
	indexWriterMu   sync.Mutex
	indexWriterStop chan struct{}
)

func startIndexWriter() {
	indexWriterMu.Lock()
	defer indexWriterMu.Unlock()
	if indexWriterStop != nil {
		return
	}
	indexWriteCh = make(chan indexWriteJob, 512)
	indexWriterStop = make(chan struct{})
	go runIndexWriter()
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
	indexWriteCh = make(chan indexWriteJob, 512)
	indexWriterStop = make(chan struct{})
	go runIndexWriter()
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

func runIndexWriter() {
	for {
		select {
		case <-indexWriterStop:
			return
		case job, ok := <-indexWriteCh:
			if !ok {
				return
			}
			job.done <- indexWriteTx(job.fn)
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
	done := make(chan error, 1)
	indexWriterMu.Lock()
	ch := indexWriteCh
	stop := indexWriterStop
	indexWriterMu.Unlock()
	if ch == nil {
		return indexWriteTx(fn)
	}
	select {
	case ch <- indexWriteJob{fn: fn, done: done}:
		return <-done
	case <-stop:
		return indexWriteTx(fn)
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
	done := make(chan error, 1)
	select {
	case ch <- indexWriteJob{fn: func(*sql.Tx) error { return nil }, done: done}:
		<-done
	case <-stop:
	}
}
