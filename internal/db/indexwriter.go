package db

import (
	"database/sql"
	"sync"
)

type indexWriteJob struct {
	fn   func(*sql.Tx) error
	done chan error
}

var (
	indexWriteCh   chan indexWriteJob
	indexWriterOnce sync.Once
	indexWriterStop chan struct{}
)

func startIndexWriter() {
	indexWriterOnce.Do(func() {
		indexWriteCh = make(chan indexWriteJob, 512)
		indexWriterStop = make(chan struct{})
		go runIndexWriter()
	})
}

func stopIndexWriter() {
	if indexWriterStop == nil {
		return
	}
	select {
	case <-indexWriterStop:
		return
	default:
		close(indexWriterStop)
	}
}

func runIndexWriter() {
	for {
		select {
		case <-indexWriterStop:
			return
		case job := <-indexWriteCh:
			job.done <- indexWriteTx(job.fn)
		}
	}
}

func indexWriteTx(fn func(*sql.Tx) error) error {
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
	startIndexWriter()
	done := make(chan error, 1)
	select {
	case indexWriteCh <- indexWriteJob{fn: fn, done: done}:
		return <-done
	case <-indexWriterStop:
		return indexWriteTx(fn)
	}
}

// FlushIndexWriter drains pending index writes (call before WAL checkpoint).
func FlushIndexWriter() {
	startIndexWriter()
	done := make(chan error, 1)
	indexWriteCh <- indexWriteJob{fn: func(*sql.Tx) error { return nil }, done: done}
	<-done
}
