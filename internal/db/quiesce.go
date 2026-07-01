package db

import (
	"database/sql"
	"fmt"
	"sync/atomic"
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
		return nil, fmt.Errorf("index db quiesced for maintenance")
	}
	if IndexDB == nil {
		return nil, fmt.Errorf("index db unavailable")
	}
	return IndexDB, nil
}

func quiesceIndexPool() error {
	indexReadGate.Store(true)
	FlushIndexWriter()
	stopIndexWriter()
	if IndexDB == nil {
		return nil
	}
	if err := IndexDB.Close(); err != nil {
		return fmt.Errorf("close index pool: %w", err)
	}
	IndexDB = nil
	return nil
}

func restoreIndexPool() error {
	if IndexDB != nil {
		indexReadGate.Store(false)
		return nil
	}
	conn, err := openPool(indexDBPath())
	if err != nil {
		indexReadGate.Store(false)
		return err
	}
	IndexDB = conn
	resetIndexWriter()
	indexReadGate.Store(false)
	return nil
}
