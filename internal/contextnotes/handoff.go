package contextnotes

import (
	"database/sql"
	"errors"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// KindHandoffResult marks a child's completion result. Such notes, like every note a handoff
// child session stores, are owned by the handoff tree: orphan purges and LRU eviction skip
// them, and the tree's expiry or flush deletes them (FlushSession).
const KindHandoffResult = "handoff_result"

// Peek returns the note stored under ref without recording an access, so copying a note into a
// handoff snapshot doesn't count as the parent fetching it back. It fails with CodeNotFound
// when no note has that ref.
func Peek(ref string) (*Note, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "ref required")
	}
	n, err := noteByRef(ref)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errs.NewCode(errs.CodeNotFound, "context note not found", "ref", ref)
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to read context note", err, "ref", ref)
	}
	return &n, nil
}

// FlushSession deletes every note stored by sessionID, including its vectors, FTS rows, and
// session stats, and returns how many were deleted. Tree expiry uses it for child sessions,
// whose notes the usual orphan and LRU paths never touch.
func FlushSession(sessionID string) (int, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return 0, errs.NewCode(errs.CodeInvalidInput, "session_id required")
	}
	_, count, err := deleteBySession(sessionID, "")
	if err != nil {
		return count, errs.WrapMessage("failed to flush session notes", err, "session_id", sessionID)
	}
	return count, nil
}
