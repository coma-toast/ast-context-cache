package db

import (
	"database/sql"
	"errors"
	"os"
	"testing"
	"time"
)

// strandTimeout bounds every wait on a write that used to be stranded forever;
// hitting it means the caller would have blocked on job.done indefinitely.
const strandTimeout = 5 * time.Second

func initIndexWriterTest(t *testing.T) {
	t.Helper()
	// Not t.TempDir(): Init's background FTS rebuild runs on the original pool
	// handle and can still be writing WAL files after Close, which makes
	// t.TempDir's strict RemoveAll fail the test. Remove best-effort instead.
	dir, err := os.MkdirTemp("", "indexwriter-test-")
	if err != nil {
		t.Fatal(err)
	}
	restoreHome := SetHomeForTest(dir)
	t.Cleanup(func() {
		indexReadGate.Store(false)
		Close()
		restoreHome()
		_ = os.RemoveAll(dir)
	})
	if err := Init(); err != nil {
		t.Fatal(err)
	}
}

func waitIndexWrite(t *testing.T, errc <-chan error, what string) error {
	t.Helper()
	select {
	case err := <-errc:
		return err
	case <-time.After(strandTimeout):
		t.Fatalf("%s still blocked after %s: write stranded on a stopped index writer", what, strandTimeout)
		return nil
	}
}

func waitBuffered(t *testing.T, ch chan *indexWriteJob, n int) {
	t.Helper()
	deadline := time.Now().Add(strandTimeout)
	for len(ch) < n {
		if time.Now().After(deadline) {
			t.Fatalf("index write channel holds %d job(s), want %d", len(ch), n)
		}
		time.Sleep(time.Millisecond)
	}
}

// The original race: IndexWrite read indexWriteCh/indexWriterStop, then
// quiesceIndexPool stopped that writer before the send. The stale channel still
// had buffer room, so select could pick the send over the closed stop, and the
// caller waited on a job nobody would ever read.
func TestSubmitIndexWriteOnStaleCaptureAfterStopReturnsError(t *testing.T) {
	initIndexWriterTest(t)

	indexWriterMu.Lock()
	ch, stop := indexWriteCh, indexWriterStop
	indexWriterMu.Unlock()
	if ch == nil || stop == nil {
		t.Fatal("expected Init to start the index writer")
	}

	stopIndexWriter()

	ran := make(chan struct{}, 1)
	errc := make(chan error, 1)
	go func() {
		errc <- submitIndexWrite(ch, stop, func(*sql.Tx) error {
			ran <- struct{}{}
			return nil
		})
	}()
	if err := waitIndexWrite(t, errc, "submit to stopped writer"); !errors.Is(err, errIndexWriterStopped) {
		t.Fatalf("submit to stopped writer: err=%v, want errIndexWriterStopped", err)
	}
	select {
	case <-ran:
		t.Fatal("write ran after its writer was stopped")
	default:
	}
}

// A job already sitting in the buffer when the writer's stop closes must be
// released, not left for a writer that has exited. No writer goroutine reads
// this channel, so the send deterministically lands before stop closes.
func TestSubmitIndexWriteBufferedJobReleasedWhenStopCloses(t *testing.T) {
	ch := make(chan *indexWriteJob, 1)
	stop := make(chan struct{})

	errc := make(chan error, 1)
	go func() {
		errc <- submitIndexWrite(ch, stop, func(*sql.Tx) error {
			t.Error("write ran with no writer")
			return nil
		})
	}()
	waitBuffered(t, ch, 1)
	close(stop)

	if err := waitIndexWrite(t, errc, "buffered write"); !errors.Is(err, errIndexWriterStopped) {
		t.Fatalf("buffered write: err=%v, want errIndexWriterStopped", err)
	}
	if job := <-ch; job.claim() {
		t.Fatal("abandoned job still claimable by a writer")
	}
}

// The same stranding through the real writer: a job enqueued behind an
// in-flight write (i.e. after FlushIndexWriter returned) when quiesce stops the
// writer. runIndexWriter's select could pick stop over the buffered job.
func TestIndexWriteBufferedBehindInFlightWriteFailsOnStop(t *testing.T) {
	initIndexWriterTest(t)

	running := make(chan struct{})
	release := make(chan struct{})
	firstc := make(chan error, 1)
	go func() {
		firstc <- IndexWrite(func(*sql.Tx) error {
			close(running)
			<-release
			return nil
		})
	}()
	<-running

	indexWriterMu.Lock()
	ch := indexWriteCh
	indexWriterMu.Unlock()

	secondRan := make(chan struct{}, 1)
	secondc := make(chan error, 1)
	go func() {
		secondc <- IndexWrite(func(*sql.Tx) error {
			secondRan <- struct{}{}
			return nil
		})
	}()
	waitBuffered(t, ch, 1)

	stopIndexWriter()
	close(release)

	if err := waitIndexWrite(t, firstc, "in-flight write"); err != nil {
		t.Fatalf("in-flight write: %v", err)
	}
	if err := waitIndexWrite(t, secondc, "buffered write"); !errors.Is(err, errIndexWriterStopped) {
		t.Fatalf("buffered write: err=%v, want errIndexWriterStopped", err)
	}
	select {
	case <-secondRan:
		t.Fatal("buffered write ran after its writer was stopped")
	default:
	}
}

// A caller that passed IndexWrite's gate check just before quiesceIndexPool
// stopped the writer must not start a fresh one against the closing pool.
func TestStartIndexWriterSkippedWhileQuiesced(t *testing.T) {
	initIndexWriterTest(t)

	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}
	startIndexWriter()
	indexWriterMu.Lock()
	restarted := indexWriterStop != nil
	indexWriterMu.Unlock()
	if restarted {
		t.Fatal("startIndexWriter started a writer while the index pool was quiesced")
	}

	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}
	if err := IndexWrite(func(tx *sql.Tx) error {
		_, err := tx.Exec(`CREATE TABLE IF NOT EXISTS writer_restart_test (id INTEGER PRIMARY KEY)`)
		return err
	}); err != nil {
		t.Fatalf("IndexWrite after restore: %v", err)
	}
}
