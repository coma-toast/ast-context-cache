package memory

import (
	"database/sql"
	"errors"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Peek returns the entry stored under ref without recording an access, so copying it into a
// handoff snapshot doesn't count as a recall. It fails with CodeNotFound when no entry has that
// ref.
func Peek(ref string) (*Entry, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "ref required")
	}
	e, err := entryByRef(ref)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errs.NewCode(errs.CodeNotFound, "memory entry not found", "ref", ref)
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to read memory entry", err, "ref", ref)
	}
	return &e, nil
}
