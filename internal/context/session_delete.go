package context

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const deleteSessionReturnedQueryPrefix = `DELETE FROM sessions WHERE session_id IN (`

// DeleteSessionKeys forgets everything returned to sids: their sessions rows in usage.db and
// their in-memory sets. Handoff tree expiry calls it for child sessions (DC-4).
//
// The write buffer is flushed first so rows enqueued before the call can't land after the
// delete; the in-memory sets go last so a call racing the delete can't rehydrate from rows
// that are about to disappear and keep them.
func DeleteSessionKeys(sids ...string) error {
	var ids []any
	for _, sid := range sids {
		if sid = strings.TrimSpace(sid); sid != "" {
			ids = append(ids, sid)
		}
	}
	if len(ids) == 0 {
		return nil
	}
	if db.DB != nil {
		db.FlushWriteBuffers()
		q := deleteSessionReturnedQueryPrefix + strings.TrimSuffix(strings.Repeat("?,", len(ids)), ",") + ")"
		if _, err := db.DB.Exec(q, ids...); err != nil {
			return errs.WrapMessage("failed to delete returned symbols", err, "sessions", len(ids))
		}
	}
	for _, sid := range ids {
		evictSession(sid.(string))
	}
	return nil
}

// evictSession drops sid's in-memory set, marking it evicted so a holder of the old pointer
// retries with a fresh one rather than writing into a dropped set.
func evictSession(sid string) {
	v, ok := sessions.LoadAndDelete(sid)
	if !ok {
		return
	}
	s := v.(*sessionSet)
	s.mu.Lock()
	s.evicted = true
	s.mu.Unlock()
}
