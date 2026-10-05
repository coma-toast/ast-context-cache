package db

import (
	"database/sql"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

var errHandoffPoolUnavailable = errs.NewCode(errs.CodeInternal, "handoff write pool unavailable")

// HandoffTx runs fn in one IMMEDIATE transaction on HandoffWriteDB and commits it, or rolls it
// back when fn returns an error or panics. fn's error is returned unwrapped so its codes and
// identity reach the caller.
//
// The pool has one connection, so concurrent callers queue for it in Go rather than spinning on
// SQLITE_BUSY, and BEGIN IMMEDIATE takes the write lock up front so a read-then-write inside fn
// sees no interleaved handoff write.
func HandoffTx(fn func(*sql.Tx) error) (err error) {
	conn := HandoffWriteDB
	if conn == nil {
		return errHandoffPoolUnavailable
	}
	tx, err := conn.Begin()
	if err != nil {
		return errs.WrapMessage("failed to begin handoff transaction", err)
	}
	defer func() {
		if p := recover(); p != nil {
			tx.Rollback()
			panic(p)
		}
	}()
	if err := fn(tx); err != nil {
		tx.Rollback()
		return err
	}
	if err := tx.Commit(); err != nil {
		return errs.WrapMessage("failed to commit handoff transaction", err)
	}
	return nil
}
