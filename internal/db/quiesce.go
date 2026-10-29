package db

import (
	"database/sql"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

var (
	errIndexQuiesced    = errs.New("index db quiesced for maintenance")
	errIndexUnavailable = errs.New("index db unavailable")
)

var indexReadGate atomic.Bool

// IndexReadQuiesced reports whether index.db reads are blocked for WAL TRUNCATE.
func IndexReadQuiesced() bool {
	return indexReadGate.Load()
}

// EmbedQueueIdleHook returns true when embed workers are idle (set from main).
var EmbedQueueIdleHook func() bool

func embedQueueIdle() bool {
	if EmbedQueueIdleHook == nil {
		return true
	}
	return EmbedQueueIdleHook()
}

func IndexReader() (*sql.DB, error) {
	if IndexReadQuiesced() {
		return nil, errIndexQuiesced
	}
	if IndexDB == nil {
		return nil, errIndexUnavailable
	}
	return IndexDB, nil
}

func quiesceIndexPool() error {
	indexReadGate.Store(true)
	FlushIndexWriter()
	stopIndexWriter()
	// Wait rather than cancel: an interrupted rebuild rolls back and isn't retried
	// until the next Init. maintainWAL skips quiescing while one is running, so this
	// normally returns at once.
	waitFTSRebuild()
	var err error
	for _, pool := range []**sql.DB{&indexWriteDB, &IndexDB} {
		if *pool == nil {
			continue
		}
		if cerr := (*pool).Close(); cerr != nil && err == nil {
			err = errs.WrapMessage("failed to close index pool", cerr)
		}
		*pool = nil
	}
	syncPoolsOpen()
	return err
}

// restoreIndexPool reopens whichever of IndexDB and indexWriteDB is closed and
// restarts the index writer, then lifts the read gate.
func restoreIndexPool() error {
	defer indexReadGate.Store(false)
	if IndexDB != nil && indexWriteDB != nil {
		return nil
	}
	path := indexDBPath()
	if IndexDB == nil {
		conn, err := openPool(path)
		if err != nil {
			return err
		}
		IndexDB = conn
		syncPoolsOpen()
	}
	if indexWriteDB == nil {
		conn, err := openIndexWritePool(path)
		if err != nil {
			return err
		}
		indexWriteDB = conn
	}
	resetIndexWriter()
	return nil
}
